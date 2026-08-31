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
//! Per layer, `k` and `v` are `[batch, n_kv_head, seq, head_dim]` — the shape the
//! attention code has right after `qkv ... transpose(1, 2)`. The cache grows
//! along `dim = 2` (the sequence axis) via [`Tensor::cat`].
//!
//! Under grouped-query attention `n_kv_head < n_head`, and it is the **narrow**
//! `n_kv_head` tensor that is cached; expanding to one key/value per query head
//! ([`crate::layers::repeat_kv`]) happens after the read, per step, and is never
//! stored. Caching the expanded copy would multiply the footprint by the group
//! count — 8x on Qwen3-8B — for no information gain.
//!
//! # Cost — and what this implementation does *not* do
//!
//! Two deliberate simplifications shape every number in `benchmarks/`:
//!
//! 1. **`append` grows the cache with [`Tensor::cat`]**, which copies the whole
//!    cache on every decode step. The step is still O(n) overall (the attention
//!    matmul is O(n·head_dim) anyway), but the copy is a large constant, and
//!    total decode work is O(n²) in *bytes moved*. A capacity-doubling
//!    preallocated buffer written in place would remove it.
//! 2. **Quantized modes dequantize the entire cache on every step**, so the
//!    attention matmul stays at the model precision. That adds a second O(n)
//!    pass per step on top of the copy.
//!
//! Together these are why cached decode throughput still falls with context
//! length, and why the quantized paths fall *faster* than fp32 (int4 more than
//! int8 — nibble unpacking is heavier than a byte cast). They also mean
//! quantization shrinks the **retained** footprint but not the transient one, so
//! peak RSS does not move. Turning the memory saving into a speed and peak-RSS
//! win requires a low-precision matmul or chunked dequantization, not a change
//! to this container.
//!
//! # Quantized storage
//!
//! With [`KvQuant::Int8`] the *stored* K/V are held as `u8` (per-token symmetric
//! int8: one fp32 scale per `(batch, head, position)`, quantizing that position's
//! `head_dim` vector to `[-127, 127]`). `append` still returns full-precision
//! tensors — it dequantizes the whole cache for the attention matmul — so the
//! error that accumulates is one int8 round-trip per stored token, and the
//! saving is in the *retained* footprint (~3.7x smaller for `gpt2`), not the
//! transient peak.
//!
//! Quantization arithmetic always runs in fp32 regardless of the model
//! precision, and the dequantized result is cast back to whatever dtype went in.
//! Computing a bf16 tensor's absolute maximum in bf16 would throw away most of
//! the range information the scale is supposed to capture.
//!
//! # Usage
//!
//! Prefill the prompt in one call, then feed a single token per decode step.
//! Position ids and the causal mask are handled by the model's
//! [`forward_with_cache`](crate::models::CausalLM::forward_with_cache).
//!
//! ```no_run
//! use candle_core::Tensor;
//! use minillm::kv_cache::KvCache;
//! use minillm::models::CausalLM;
//! use minillm::{device, loader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let dev = device::best();
//! let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;
//! let ids = tokenizer.encode("Hello", true).unwrap().get_ids().to_vec();
//!
//! let mut cache = model.new_cache();
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

/// Sequence axis for the cached `[batch, n_kv_head, seq, head_dim]` tensors.
const SEQ_DIM: usize = 2;

/// How the cached K/V are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Store K/V at the model's own precision. Default.
    #[default]
    None,
    /// Per-token symmetric int8 (`u8` storage + one fp32 scale per
    /// `(batch, head, position)`). ~3.7x smaller retained footprint for `gpt2`.
    Int8,
    /// Per-token asymmetric int4, two values packed per `u8`, with an fp32
    /// scale and zero-point per `(batch, head, position)`. ~6.4x smaller.
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

    /// Lower-case name used in CSV output and CLI arguments.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    /// Bytes of *stored* cache per sequence position.
    ///
    /// `n_kv_head` is the key/value head count — under GQA that is smaller than
    /// the query head count, and it is the one that determines cache size.
    /// `elem_bytes` is the model precision's element width, used only by
    /// [`Self::None`]; the quantized modes store the values narrower and add one
    /// fp32 scale per `(layer, kv_head, position, K|V)`, and [`Self::Int4`] also
    /// adds a zero-point.
    pub fn bytes_per_token(
        self,
        n_layer: usize,
        n_kv_head: usize,
        head_dim: usize,
        elem_bytes: usize,
    ) -> usize {
        let elems = 2 * n_layer * n_kv_head * head_dim; // K + V
        let scales = 2 * n_layer * n_kv_head * 4; // fp32 scale per (layer, head), K + V
        match self {
            Self::None => elems * elem_bytes,
            Self::Int8 => elems + scales,
            Self::Int4 => elems / 2 + 2 * scales, // 4-bit values + scale + zero-point
        }
    }
}

impl std::fmt::Display for KvQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One K or V tensor's storage: full precision, per-token int8, or per-token
/// packed int4.
///
/// The quantized variants remember the dtype they were handed so
/// [`Slot::dequantized`] can hand back the model's own precision rather than
/// silently promoting the whole cache to fp32.
#[derive(Debug)]
enum Slot {
    /// Stored at the model precision, whatever that is.
    Full(Tensor),
    /// `q`: `u8` `[b, n_kv_head, seq, head_dim]` (int8 biased by +128).
    /// `scale`: `f32` `[b, n_kv_head, seq, 1]`.
    Q8 {
        q: Tensor,
        scale: Tensor,
        dtype: DType,
    },
    /// `q`: `u8` `[b, n_kv_head, seq, head_dim / 2]` (two 0..15 nibbles per byte,
    /// low nibble = even index). `scale`, `zero`: `f32` `[b, n_kv_head, seq, 1]`.
    Q4 {
        q: Tensor,
        scale: Tensor,
        zero: Tensor,
        dtype: DType,
    },
}

impl Slot {
    fn seq_len(&self) -> usize {
        let t = match self {
            Slot::Full(t) => t,
            Slot::Q8 { q, .. } => q,
            Slot::Q4 { q, .. } => q,
        };
        t.dims().get(SEQ_DIM).copied().unwrap_or(0)
    }

    /// Full-precision view of what is stored, in the dtype it was appended at.
    fn dequantized(&self) -> Result<Tensor> {
        match self {
            Slot::Full(t) => Ok(t.clone()),
            Slot::Q8 { q, scale, dtype } => dequantize_i8(q, scale, *dtype),
            Slot::Q4 {
                q,
                scale,
                zero,
                dtype,
            } => dequantize_i4(q, scale, zero, *dtype),
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
    /// An empty cache that stores K/V at the model precision.
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
    /// (`[batch, n_kv_head, seq, head_dim]`). `None` means nothing is cached yet;
    /// `Some(Err(_))` is a genuine dequantization failure, kept distinct from
    /// "empty" rather than swallowed.
    pub fn keys(&self) -> Option<Result<Tensor>> {
        self.k.as_ref().map(Slot::dequantized)
    }

    /// The cached values so far, dequantized, if any. See [`keys`](Self::keys)
    /// for what `None` vs. `Some(Err(_))` mean.
    pub fn values(&self) -> Option<Result<Tensor>> {
        self.v.as_ref().map(Slot::dequantized)
    }

    /// Append this step's `k_new` / `v_new` (each `[batch, n_kv_head, s_new,
    /// head_dim]`) to the history and return the **full**, full-precision
    /// `(keys, values)` to attend the new queries over.
    ///
    /// On the first call the inputs seed the cache; afterwards they are
    /// concatenated along the sequence axis. With a quantized mode the stored
    /// copy is narrow; the returned tensors are always at the appended dtype and
    /// contiguous.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let quant = self.quant;
        let k = push(&mut self.k, quant, k_new)?;
        let v = push(&mut self.v, quant, v_new)?;
        Ok((k, v))
    }
}

/// Extend `slot` with `new` under `quant`, returning the full dequantized tensor.
fn push(slot: &mut Option<Slot>, quant: KvQuant, new: &Tensor) -> Result<Tensor> {
    let new = new.contiguous()?;
    let dtype = new.dtype();
    match quant {
        KvQuant::None => {
            let full = match slot.take() {
                None => new,
                Some(Slot::Full(prev)) => Tensor::cat(&[&prev, &new], SEQ_DIM)?.contiguous()?,
                Some(_) => return Err(mode_switch_err()),
            };
            *slot = Some(Slot::Full(full.clone()));
            Ok(full)
        }
        KvQuant::Int8 => {
            let (q_new, s_new) = quantize_i8_per_token(&new)?;
            let (q, scale) = match slot.take() {
                None => (q_new, s_new),
                Some(Slot::Q8 { q, scale, .. }) => (
                    Tensor::cat(&[&q, &q_new], SEQ_DIM)?.contiguous()?,
                    Tensor::cat(&[&scale, &s_new], SEQ_DIM)?.contiguous()?,
                ),
                Some(_) => return Err(mode_switch_err()),
            };
            let full = dequantize_i8(&q, &scale, dtype)?;
            *slot = Some(Slot::Q8 { q, scale, dtype });
            Ok(full)
        }
        KvQuant::Int4 => {
            let (q_new, s_new, z_new) = quantize_i4_per_token(&new)?;
            let (q, scale, zero) = match slot.take() {
                None => (q_new, s_new, z_new),
                Some(Slot::Q4 { q, scale, zero, .. }) => (
                    Tensor::cat(&[&q, &q_new], SEQ_DIM)?.contiguous()?,
                    Tensor::cat(&[&scale, &s_new], SEQ_DIM)?.contiguous()?,
                    Tensor::cat(&[&zero, &z_new], SEQ_DIM)?.contiguous()?,
                ),
                Some(_) => return Err(mode_switch_err()),
            };
            let full = dequantize_i4(&q, &scale, &zero, dtype)?;
            *slot = Some(Slot::Q4 {
                q,
                scale,
                zero,
                dtype,
            });
            Ok(full)
        }
    }
}

fn mode_switch_err() -> candle_core::Error {
    candle_core::Error::Msg("KV cache quantization mode changed mid-sequence".into())
}

/// Per-token symmetric int8. `x`: `[b, n_kv_head, seq, head_dim]`, any float
/// dtype. Returns `(q_u8 [b,h,s,hd], scale_f32 [b,h,s,1])` where
/// `x ≈ (q_u8 - 128) * scale`.
///
/// The scale search and the division run in fp32 even for a bf16 model: bf16 has
/// 8 mantissa bits, so computing the row maximum in it would quantize the scale
/// itself before it is ever applied.
fn quantize_i8_per_token(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let x = x.to_dtype(DType::F32)?;
    let amax = x.abs()?.max_keepdim(D::Minus1)?; // [b,h,s,1]
                                                 // +eps so an all-zero row still gets a finite, invertible scale.
    let scale = ((amax + 1e-9)? / 127.0)?;
    let q = x.broadcast_div(&scale)?.round()?.clamp(-127f32, 127f32)?;
    let q_u8 = (q + 128.0)?.to_dtype(DType::U8)?;
    Ok((q_u8, scale))
}

/// Inverse of [`quantize_i8_per_token`], cast back to `dtype`.
fn dequantize_i8(q_u8: &Tensor, scale: &Tensor, dtype: DType) -> Result<Tensor> {
    let q = (q_u8.to_dtype(DType::F32)? - 128.0)?;
    q.broadcast_mul(scale)?.to_dtype(dtype)?.contiguous()
}

/// Per-token asymmetric int4, two values packed per byte. `x`: `[b, n_kv_head,
/// seq, head_dim]`, any float dtype, `head_dim` even. Returns
/// `(q_u8 [b,h,s,hd/2], scale [b,h,s,1], zero [b,h,s,1])` with
/// `x ≈ q * scale + zero` for the unpacked `q` in `0..=15` (low nibble = even
/// index).
fn quantize_i4_per_token(x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, h, s, hd) = x.dims4()?;
    if hd % 2 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "int4 KV cache needs an even head_dim, got {hd}"
        )));
    }
    let x = x.to_dtype(DType::F32)?;

    let zero = x.min_keepdim(D::Minus1)?; // [b,h,s,1]
    let xmax = x.max_keepdim(D::Minus1)?;
    // +eps so a constant row still gets a finite, invertible scale.
    let scale = ((xmax.broadcast_sub(&zero)? + 1e-9)? / 15.0)?;

    let q = x
        .broadcast_sub(&zero)?
        .broadcast_div(&scale)?
        .round()?
        .clamp(0f32, 15f32)?;

    // Pack (even, odd) nibble pairs: byte = lo + 16 * hi.
    let q = q.reshape((b, h, s, hd / 2, 2))?;
    let lo = q.narrow(4, 0, 1)?.squeeze(4)?;
    let hi = q.narrow(4, 1, 1)?.squeeze(4)?;
    let packed = (lo + (hi * 16.0)?)?.to_dtype(DType::U8)?;
    Ok((packed, scale, zero))
}

/// Inverse of [`quantize_i4_per_token`], cast back to `dtype`.
fn dequantize_i4(packed: &Tensor, scale: &Tensor, zero: &Tensor, dtype: DType) -> Result<Tensor> {
    let (b, h, s, hp) = packed.dims4()?;
    let p = packed.to_dtype(DType::F32)?;
    let hi = (p.clone() / 16.0)?.floor()?;
    let lo = (p - (hi.clone() * 16.0)?)?;
    let q = Tensor::stack(&[&lo, &hi], 4)?.reshape((b, h, s, hp * 2))?;
    q.broadcast_mul(scale)?
        .broadcast_add(zero)?
        .to_dtype(dtype)?
        .contiguous()
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
            layers: (0..n_layer)
                .map(|_| LayerKvCache::with_quant(quant))
                .collect(),
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

    /// `[batch=1, n_kv_head=2, seq, head_dim=4]` filled with `fill`.
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
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
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
        assert_eq!(c.keys().unwrap().unwrap().dims(), &[1, 2, 9, 4]);
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
        assert_eq!(
            LayerKvCache::with_quant(KvQuant::Int8).quant(),
            KvQuant::Int8
        );
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

        assert!(max_abs_diff(&k_one, &inc.keys().unwrap().unwrap()) < 1e-5);
    }

    #[test]
    fn int4_append_tracks_length_and_returns_full_precision() {
        let mut c = LayerKvCache::with_quant(KvQuant::Int4);
        c.append(&kv_ramp(6), &kv_ramp(6)).unwrap();
        assert_eq!(c.len(), 6);
        let (k, _) = c.append(&kv_ramp(1), &kv_ramp(1)).unwrap();
        assert_eq!(c.len(), 7);
        assert_eq!(k.dims(), &[1, 2, 7, 4]);
        assert_eq!(k.dtype(), DType::F32);
    }

    #[test]
    fn int4_roundtrip_within_step_size() {
        let mut c = LayerKvCache::with_quant(KvQuant::Int4);
        let x = kv_ramp(10);
        let (k, _) = c.append(&x, &x).unwrap();

        let vals = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (lo, hi) = vals
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        // per-token asymmetric int4: worst-case error is one step, (range)/15.
        assert!(
            max_abs_diff(&x, &k) <= (hi - lo) / 15.0 + 1e-4,
            "int4 round-trip error {} exceeded {}",
            max_abs_diff(&x, &k),
            (hi - lo) / 15.0
        );
    }

    #[test]
    fn int4_incremental_matches_one_shot() {
        let x = kv_ramp(5);

        let mut inc = LayerKvCache::with_quant(KvQuant::Int4);
        for t in 0..5 {
            let xt = x.narrow(SEQ_DIM, t, 1).unwrap();
            inc.append(&xt, &xt).unwrap();
        }
        let mut one = LayerKvCache::with_quant(KvQuant::Int4);
        let (k_one, _) = one.append(&x, &x).unwrap();

        assert!(max_abs_diff(&k_one, &inc.keys().unwrap().unwrap()) < 1e-5);
    }

    #[test]
    fn bytes_per_token_matches_the_documented_gpt2_figures() {
        // gpt2-124M: n_layer=12, n_kv_head=12, head_dim=64, fp32.
        let (nl, nh, hd) = (12, 12, 64);
        assert_eq!(
            KvQuant::None.bytes_per_token(nl, nh, hd, 4),
            2 * 12 * 768 * 4
        );
        assert_eq!(
            KvQuant::Int8.bytes_per_token(nl, nh, hd, 4),
            2 * 12 * 768 + 2 * 12 * 12 * 4
        );
        assert_eq!(
            KvQuant::Int4.bytes_per_token(nl, nh, hd, 4),
            2 * 12 * 768 / 2 + 2 * (2 * 12 * 12 * 4)
        );
        // int8 < fp32/3, int4 < int8
        assert!(
            KvQuant::Int8.bytes_per_token(nl, nh, hd, 4) * 3
                < KvQuant::None.bytes_per_token(nl, nh, hd, 4)
        );
        assert!(
            KvQuant::Int4.bytes_per_token(nl, nh, hd, 4)
                < KvQuant::Int8.bytes_per_token(nl, nh, hd, 4)
        );
    }

    #[test]
    fn only_the_unquantized_mode_depends_on_model_precision() {
        let (nl, nh, hd) = (28, 8, 128);
        // fp32 -> bf16 halves the stored cache...
        assert_eq!(
            KvQuant::None.bytes_per_token(nl, nh, hd, 2) * 2,
            KvQuant::None.bytes_per_token(nl, nh, hd, 4)
        );
        // ...but int8/int4 store fixed-width values, so precision is irrelevant.
        assert_eq!(
            KvQuant::Int8.bytes_per_token(nl, nh, hd, 2),
            KvQuant::Int8.bytes_per_token(nl, nh, hd, 4)
        );
        assert_eq!(
            KvQuant::Int4.bytes_per_token(nl, nh, hd, 2),
            KvQuant::Int4.bytes_per_token(nl, nh, hd, 4)
        );
    }

    #[test]
    fn bf16_round_trips_at_bf16_not_f32() {
        // A bf16 model must get a bf16 cache back: silently returning fp32 would
        // make every downstream matmul fail on a dtype mismatch.
        let x = kv_ramp(4).to_dtype(DType::BF16).unwrap();
        for quant in [KvQuant::None, KvQuant::Int8, KvQuant::Int4] {
            let mut c = LayerKvCache::with_quant(quant);
            let (k, v) = c.append(&x, &x).unwrap();
            assert_eq!(k.dtype(), DType::BF16, "{quant} keys");
            assert_eq!(v.dtype(), DType::BF16, "{quant} values");
            assert_eq!(c.keys().unwrap().unwrap().dtype(), DType::BF16);
        }
    }

    #[test]
    fn bf16_int8_quantization_stays_accurate() {
        // Quantization math runs in fp32; the only loss should be the int8 step
        // plus one bf16 rounding, not a bf16-computed scale.
        let x = kv_ramp(8).to_dtype(DType::BF16).unwrap();
        let mut c = LayerKvCache::with_quant(KvQuant::Int8);
        let (k, _) = c.append(&x, &x).unwrap();

        let peak = x
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        // bf16 has ~3 decimal digits, so allow one bf16 ulp on top of the int8 step.
        assert!(
            max_abs_diff(&x, &k) <= peak / 127.0 + peak / 256.0 + 1e-4,
            "bf16 int8 round-trip error {}",
            max_abs_diff(&x, &k)
        );
    }

    #[test]
    fn mixing_dtypes_mid_sequence_does_not_corrupt_the_cache() {
        // Appending f32 after bf16 changes the reported dtype rather than
        // silently reinterpreting the stored bytes.
        let mut c = LayerKvCache::with_quant(KvQuant::Int8);
        let bf = kv_ramp(2).to_dtype(DType::BF16).unwrap();
        c.append(&bf, &bf).unwrap();
        let f = kv_ramp(1);
        let (k, _) = c.append(&f, &f).unwrap();
        assert_eq!(k.dtype(), DType::F32);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn switching_quant_mode_mid_sequence_is_an_error() {
        let mut c = LayerKvCache::with_quant(KvQuant::None);
        c.append(&kv_ramp(2), &kv_ramp(2)).unwrap();
        c.quant = KvQuant::Int8;
        assert!(c.append(&kv_ramp(1), &kv_ramp(1)).is_err());
    }

    #[test]
    fn quant_parses_and_displays() {
        assert_eq!(KvQuant::parse("INT8"), Some(KvQuant::Int8));
        assert_eq!(KvQuant::parse("off"), Some(KvQuant::None));
        assert_eq!(KvQuant::parse("int3"), None);
        assert_eq!(KvQuant::Int4.to_string(), "int4");
    }
}
