//! Key/value cache for autoregressive decoding.
//!
//! Without a cache, every decode step re-runs the whole prefix through all
//! layers: attention is recomputed over `[seq, seq]` each time, so decoding is
//! O(n^2) in sequence length. A KV cache keeps each layer's projected keys and
//! values around, so a decode step only projects the *one* new token and
//! attends it over the stored history — O(n) per step.
//!
//! # Layout
//!
//! Per layer, `k` and `v` are `[batch, n_head, seq, head_dim]` — the shape the
//! attention code already has right after `qkv ... transpose(1, 2)`. The cache
//! grows along `dim = 2` (the sequence axis) via [`Tensor::cat`].
//!
//! [`LayerKvCache::append`] concatenates the new step's K/V onto the history and
//! returns the **full** K/V, so the caller attends the new queries over
//! everything seen so far.
//!
//! # Usage
//!
//! Prefill the prompt in one call, then feed a single token per decode step.
//! Position ids and the causal mask are handled by
//! [`GPT2Model::forward_with_cache`](crate::model::GPT2Model::forward_with_cache).
//!
//! ```no_run
//! use candle_core::Tensor;
//! use minillm::kv_cache::KvCache;
//! use minillm::{device, loader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let dev = device::best();
//! let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;
//! let ids = tokenizer.encode("Hello", true).unwrap().get_ids().to_vec();
//!
//! let mut cache = KvCache::new(model.config().n_layer);
//! let prompt = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
//! let mut logits = model.forward_with_cache(&prompt, &mut cache)?; // prefill
//!
//! let step = Tensor::from_vec(vec![42u32], (1, 1), &dev)?;
//! logits = model.forward_with_cache(&step, &mut cache)?;           // O(n) decode
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```

use candle_core::{Result, Tensor};

/// Sequence axis for the cached `[batch, n_head, seq, head_dim]` tensors.
const SEQ_DIM: usize = 2;

/// How the cached K/V are stored. `None` is plain fp32; the quantized variants
/// are the planned KV-cache-quantization ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Store K/V at the model dtype (fp32 here). Default.
    #[default]
    None,
    /// Symmetric per-head int8 with an fp32 scale per head.
    Int8,
    /// Packed int4 (two values per byte), per-head scale (+ zero-point).
    Int4,
}

/// One transformer layer's cached keys and values.
///
/// Empty until the first [`append`](Self::append). Both tensors always share the
/// same shape except along [`SEQ_DIM`].
#[derive(Debug, Default)]
pub struct LayerKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    quant: KvQuant,
}

impl LayerKvCache {
    /// An empty cache that stores K/V at full precision.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache that will quantize stored K/V with `quant`.
    ///
    /// The quantized read/write paths are not implemented yet; constructing with
    /// a non-`None` mode is accepted so callers and the plan can reference it,
    /// but [`append`](Self::append) currently always stores full precision.
    pub fn with_quant(quant: KvQuant) -> Self {
        Self {
            k: None,
            v: None,
            quant,
        }
    }

    /// Quantization mode for this layer's cache.
    pub fn quant(&self) -> KvQuant {
        self.quant
    }

    /// Number of cached positions (0 before the first append).
    pub fn len(&self) -> usize {
        match &self.k {
            Some(t) => t.dims().get(SEQ_DIM).copied().unwrap_or(0),
            None => 0,
        }
    }

    /// Whether anything is cached yet.
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    /// Drop the cached tensors; [`len`](Self::len) returns to 0.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// The cached keys so far, if any (`[batch, n_head, seq, head_dim]`).
    pub fn keys(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// The cached values so far, if any (`[batch, n_head, seq, head_dim]`).
    pub fn values(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }

    /// Append this step's `k_new` / `v_new` (each `[batch, n_head, s_new,
    /// head_dim]`) to the history and return the **full** `(keys, values)` to
    /// attend the new queries over.
    ///
    /// On the first call the inputs become the cache as-is; afterwards they are
    /// concatenated along the sequence axis. The returned tensors are
    /// contiguous so a following `matmul` is happy.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            None => k_new.contiguous()?,
            Some(prev) => Tensor::cat(&[prev, k_new], SEQ_DIM)?.contiguous()?,
        };
        let v = match &self.v {
            None => v_new.contiguous()?,
            Some(prev) => Tensor::cat(&[prev, v_new], SEQ_DIM)?.contiguous()?,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

/// The whole model's cache: one [`LayerKvCache`] per transformer block.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    /// A cache for a model with `n_layer` transformer blocks, full precision.
    pub fn new(n_layer: usize) -> Self {
        Self {
            layers: (0..n_layer).map(|_| LayerKvCache::new()).collect(),
        }
    }

    /// A cache whose every layer quantizes stored K/V with `quant`.
    pub fn with_quant(n_layer: usize, quant: KvQuant) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| LayerKvCache::with_quant(quant))
                .collect(),
        }
    }

    /// Number of transformer layers this cache covers.
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// Cached sequence length. Every layer advances together during a forward
    /// pass, so layer 0's length is the model's cached position count. Returns 0
    /// for a fresh cache.
    pub fn len(&self) -> usize {
        self.layers.first().map(LayerKvCache::len).unwrap_or(0)
    }

    /// Whether nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear every layer; [`len`](Self::len) returns to 0. Reuse across
    /// generations instead of allocating a new cache.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Mutable handle to layer `i`'s cache.
    ///
    /// # Panics
    /// If `i >= n_layer()`.
    pub fn layer(&mut self, i: usize) -> &mut LayerKvCache {
        &mut self.layers[i]
    }

    /// Non-panicking variant of [`layer`](Self::layer).
    pub fn get(&mut self, i: usize) -> Option<&mut LayerKvCache> {
        self.layers.get_mut(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};

    /// `[batch=1, n_head=2, seq, head_dim=4]` filled with `fill`.
    fn kv(seq: usize, fill: f32) -> Tensor {
        Tensor::full(fill, (1usize, 2, seq, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    #[test]
    fn layer_starts_empty() {
        let c = LayerKvCache::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert!(c.keys().is_none());
    }

    #[test]
    fn append_grows_along_seq_dim() {
        let mut c = LayerKvCache::new();

        let (k, v) = c.append(&kv(5, 1.0), &kv(5, 2.0)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 5, 4]);
        assert_eq!(v.dims(), &[1, 2, 5, 4]);
        assert_eq!(c.len(), 5);

        let (k, _) = c.append(&kv(1, 1.0), &kv(1, 2.0)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 6, 4]);
        assert_eq!(c.len(), 6);

        // three more decode steps
        for _ in 0..3 {
            c.append(&kv(1, 1.0), &kv(1, 2.0)).unwrap();
        }
        assert_eq!(c.len(), 9);
        assert_eq!(c.keys().unwrap().dims(), &[1, 2, 9, 4]);
    }

    #[test]
    fn appended_contents_are_preserved_in_order() {
        let mut c = LayerKvCache::new();
        c.append(&kv(2, 7.0), &kv(2, 0.0)).unwrap();
        let (k, _) = c.append(&kv(1, 9.0), &kv(1, 0.0)).unwrap();
        let rows = k.i((0, 0)).unwrap().to_vec2::<f32>().unwrap(); // [seq, head_dim]
        assert_eq!(rows[0], vec![7.0; 4]);
        assert_eq!(rows[1], vec![7.0; 4]);
        assert_eq!(rows[2], vec![9.0; 4]);
    }

    #[test]
    fn reset_clears_layer() {
        let mut c = LayerKvCache::new();
        c.append(&kv(4, 1.0), &kv(4, 1.0)).unwrap();
        assert_eq!(c.len(), 4);
        c.reset();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn model_cache_tracks_length_and_layers() {
        let mut cache = KvCache::new(12);
        assert_eq!(cache.n_layer(), 12);
        assert!(cache.is_empty());

        // simulate a prefill of 8 tokens across every layer
        for i in 0..12 {
            cache.layer(i).append(&kv(8, 1.0), &kv(8, 1.0)).unwrap();
        }
        assert_eq!(cache.len(), 8);

        // one decode step
        for i in 0..12 {
            cache.layer(i).append(&kv(1, 1.0), &kv(1, 1.0)).unwrap();
        }
        assert_eq!(cache.len(), 9);

        cache.reset();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn quant_mode_is_carried() {
        let cache = KvCache::with_quant(4, KvQuant::Int8);
        assert_eq!(cache.n_layer(), 4);
        let layer = LayerKvCache::with_quant(KvQuant::Int4);
        assert_eq!(layer.quant(), KvQuant::Int4);
        assert_eq!(LayerKvCache::new().quant(), KvQuant::None);
    }

    #[test]
    fn get_is_non_panicking() {
        let mut cache = KvCache::new(2);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
    }
}
