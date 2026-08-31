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
//! returns the **full**, full-precision K/V, so the caller attends the new
//! queries over everything seen so far.
//!
//! # Quantized storage
//!
//! With [`KvQuant::Int8`] the *stored* K/V are held as `u8` (per-token symmetric
//! int8: one fp32 scale per `(batch, head, position)`, quantizing that position's
//! `head_dim` vector to `[-127, 127]`). `append` still returns full-precision
//! tensors — it dequantizes the whole cache for the attention matmul — so the
//! error that accumulates is one int8 round-trip per stored token, and the
//! saving is in the *retained* footprint (~3.7× smaller for `gpt2`), not the
//! transient peak.
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

use candle_core::{DType, Result, Tensor, D};

/// Sequence axis for the cached `[batch, n_head, seq, head_dim]` tensors.
const SEQ_DIM: usize = 2;

/// How the cached K/V are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Store K/V at the model dtype (fp32 here). Default.
    #[default]
    None,
    /// Per-token symmetric int8 (`u8` storage + one fp32 scale per
    /// `(batch, head, position)`). ~3.7× smaller retained footprint for `gpt2`.
    Int8,
    /// Packed int4. Not implemented yet.
    Int4,
}

impl KvQuant {
    /// Parse `"none"` / `"int8"` / `"int4"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "fp32" | "f32" => Some(Self::None),
            "int8" | "i8" | "8" => Some(Self::Int8),
            "int4" | "i4" | "4" => Some(Self::Int4),
            _ => None,
        }
    }

    /// Bytes of *stored* cache per sequence position, given the model dims.
    /// `None` is `2 * n_layer * n_embd * 4`; the quantized modes add a per-head
    /// fp32 scale (`n_head` per layer per K/V).
    pub fn bytes_per_token(self, n_layer: usize, n_head: usize, head_dim: usize) -> usize {
        let elems = 2 * n_layer * n_head * head_dim; // K + V
        let scales = 2 * n_layer * n_head * 4; // fp32 scale per (layer, head), K + V
        match self {
            Self::None => elems * 4,
            Self::Int8 => elems + scales,
            Self::Int4 => elems / 2 + scales,
        }
    }
}

/// One K or V tensor's storage: full precision, or per-token int8.
#[derive(Debug)]
enum Slot {
    F32(Tensor),
    /// `q`: `u8` `[b, n_head, seq, head_dim]` (int8 biased by +128).
    /// `scale`: `f32` `[b, n_head, seq, 1]`.
    Q8 { q: Tensor, scale: Tensor },
}

impl Slot {
    fn seq_len(&self) -> usize {
        let t = match self {
            Slot::F32(t) => t,
            Slot::Q8 { q, .. } => q,
        };
        t.dims().get(SEQ_DIM).copied().unwrap_or(0)
    }

    /// Full-precision view of what is stored.
    fn to_f32(&self) -> Result<Tensor> {
        match self {
            Slot::F32(t) => Ok(t.clone()),
            Slot::Q8 { q, scale } => dequantize_i8(q, scale),
        }
    }
}

/// One transformer layer's cached keys and values.
///
/// Empty until the first [`append`](Self::append).
#[derive(Debug, Default)]
pub struct LayerKvCache {
    k: Option<Slot>,
    v: Option<Slot>,
    quant: KvQuant,
}

impl LayerKvCache {
    /// An empty cache that stores K/V at full precision.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache that quantizes stored K/V with `quant`.
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
        self.k.as_ref().map(Slot::seq_len).unwrap_or(0)
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

    /// The cached keys so far, dequantized, if any
    /// (`[batch, n_head, seq, head_dim]`).
    pub fn keys(&self) -> Option<Tensor> {
        self.k.as_ref().and_then(|s| s.to_f32().ok())
    }

    /// The cached values so far, dequantized, if any.
    pub fn values(&self) -> Option<Tensor> {
        self.v.as_ref().and_then(|s| s.to_f32().ok())
    }

    /// Append this step's `k_new` / `v_new` (each `[batch, n_head, s_new,
    /// head_dim]`) to the history and return the **full**, full-precision
    /// `(keys, values)` to attend the new queries over.
    ///
    /// On the first call the inputs seed the cache; afterwards they are
    /// concatenated along the sequence axis. With a quantized mode the stored
    /// copy is int8; the returned tensors are always fp32 and contiguous.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let quant = self.quant;
        let k = push(&mut self.k, quant, k_new)?;
        let v = push(&mut self.v, quant, v_new)?;
        Ok((k, v))
    }
}

/// Extend `slot` with `new` under `quant`, returning the full fp32 tensor.
fn push(slot: &mut Option<Slot>, quant: KvQuant, new: &Tensor) -> Result<Tensor> {
    let new = new.contiguous()?;
    match quant {
        KvQuant::None => {
            let full = match slot.take() {
                None => new,
                Some(Slot::F32(prev)) => Tensor::cat(&[&prev, &new], SEQ_DIM)?.contiguous()?,
                Some(Slot::Q8 { .. }) => return Err(mode_switch_err()),
            };
            *slot = Some(Slot::F32(full.clone()));
            Ok(full)
        }
        KvQuant::Int8 => {
            let (q_new, s_new) = quantize_i8_per_token(&new)?;
            let (q, scale) = match slot.take() {
                None => (q_new, s_new),
                Some(Slot::Q8 { q, scale }) => (
                    Tensor::cat(&[&q, &q_new], SEQ_DIM)?.contiguous()?,
                    Tensor::cat(&[&scale, &s_new], SEQ_DIM)?.contiguous()?,
                ),
                Some(Slot::F32(_)) => return Err(mode_switch_err()),
            };
            let full = dequantize_i8(&q, &scale)?;
            *slot = Some(Slot::Q8 { q, scale });
            Ok(full)
        }
        KvQuant::Int4 => Err(candle_core::Error::Msg(
            "KvQuant::Int4 is not implemented yet".into(),
        )),
    }
}

fn mode_switch_err() -> candle_core::Error {
    candle_core::Error::Msg("KV cache quantization mode changed mid-sequence".into())
}

/// Per-token symmetric int8. `x`: `[b, n_head, seq, head_dim]` fp32.
/// Returns `(q_u8 [b,h,s,hd], scale_f32 [b,h,s,1])` where
/// `x ≈ (q_u8 - 128) * scale`.
fn quantize_i8_per_token(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let amax = x.abs()?.max_keepdim(D::Minus1)?; // [b,h,s,1]
    // +eps so an all-zero row still gets a finite, invertible scale.
    let scale = ((amax + 1e-9)? / 127.0)?;
    let q = x
        .broadcast_div(&scale)?
        .round()?
        .clamp(-127f32, 127f32)?;
    let q_u8 = (q + 128.0)?.to_dtype(DType::U8)?;
    Ok((q_u8, scale))
}

/// Inverse of [`quantize_i8_per_token`].
fn dequantize_i8(q_u8: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let q = (q_u8.to_dtype(DType::F32)? - 128.0)?;
    q.broadcast_mul(scale)?.contiguous()
}

/// The whole model's cache: one [`LayerKvCache`] per transformer block.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    quant: KvQuant,
}

impl KvCache {
    /// A cache for a model with `n_layer` transformer blocks, full precision.
    pub fn new(n_layer: usize) -> Self {
        Self::with_quant(n_layer, KvQuant::None)
    }

    /// A cache whose every layer quantizes stored K/V with `quant`.
    pub fn with_quant(n_layer: usize, quant: KvQuant) -> Self {
        Self {
            layers: (0..n_layer).map(|_| LayerKvCache::with_quant(quant)).collect(),
            quant,
        }
    }

    /// Quantization mode every layer was built with.
    pub fn quant(&self) -> KvQuant {
        self.quant
    }

    /// Number of transformer layers this cache covers.
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// Cached sequence length (layer 0 — every layer advances together).
    pub fn len(&self) -> usize {
        self.layers.first().map(LayerKvCache::len).unwrap_or(0)
    }

    /// Whether nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear every layer; [`len`](Self::len) returns to 0.
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
    use candle_core::{Device, IndexOp};

    /// `[batch=1, n_head=2, seq, head_dim=4]` filled with `fill`.
    fn kv(seq: usize, fill: f32) -> Tensor {
        Tensor::full(fill, (1usize, 2, seq, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    /// `[1, 2, seq, 4]` with distinct per-element values.
    fn kv_ramp(seq: usize) -> Tensor {
        let n = 2 * seq * 4;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 1.3).collect();
        Tensor::from_vec(data, (1usize, 2, seq, 4), &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
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
        let rows = k.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
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

        for i in 0..12 {
            cache.layer(i).append(&kv(8, 1.0), &kv(8, 1.0)).unwrap();
        }
        assert_eq!(cache.len(), 8);

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
        assert_eq!(cache.quant(), KvQuant::Int8);
        assert_eq!(LayerKvCache::with_quant(KvQuant::Int8).quant(), KvQuant::Int8);
        assert_eq!(LayerKvCache::new().quant(), KvQuant::None);
    }

    #[test]
    fn get_is_non_panicking() {
        let mut cache = KvCache::new(2);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
    }

    #[test]
    fn int8_append_tracks_length() {
        let mut c = LayerKvCache::with_quant(KvQuant::Int8);
        c.append(&kv_ramp(6), &kv_ramp(6)).unwrap();
        assert_eq!(c.len(), 6);
        let (k, _) = c.append(&kv_ramp(1), &kv_ramp(1)).unwrap();
        assert_eq!(c.len(), 7);
        assert_eq!(k.dims(), &[1, 2, 7, 4]);
        assert_eq!(k.dtype(), DType::F32); // returns dequantized
    }

    #[test]
    fn int8_roundtrip_is_close() {
        let mut c = LayerKvCache::with_quant(KvQuant::Int8);
        let x = kv_ramp(10);
        let (k, _) = c.append(&x, &x).unwrap();

        let peak = x
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        // per-token symmetric int8: worst-case error is ~peak/127.
        assert!(
            max_abs_diff(&x, &k) <= peak / 127.0 + 1e-4,
            "int8 round-trip error {} exceeded {}",
            max_abs_diff(&x, &k),
            peak / 127.0
        );
    }

    #[test]
    fn int8_incremental_matches_one_shot() {
        // Per-token quant is position-independent: appending one token at a time
        // must dequantize to the same values as a single bulk append.
        let x = kv_ramp(5);

        let mut inc = LayerKvCache::with_quant(KvQuant::Int8);
        for t in 0..5 {
            let xt = x.narrow(SEQ_DIM, t, 1).unwrap();
            inc.append(&xt, &xt).unwrap();
        }

        let mut one = LayerKvCache::with_quant(KvQuant::Int8);
        let (k_one, _) = one.append(&x, &x).unwrap();

        assert!(max_abs_diff(&k_one, &inc.keys().unwrap()) < 1e-5);
    }

    #[test]
    fn bytes_per_token_matches_config_formula() {
        // gpt2-124M: n_layer=12, n_head=12, head_dim=64
        assert_eq!(KvQuant::None.bytes_per_token(12, 12, 64), 2 * 12 * 768 * 4);
        assert_eq!(
            KvQuant::Int8.bytes_per_token(12, 12, 64),
            2 * 12 * 768 + 2 * 12 * 12 * 4
        );
        assert!(
            KvQuant::Int8.bytes_per_token(12, 12, 64) * 3
                < KvQuant::None.bytes_per_token(12, 12, 64)
        );
    }
}
