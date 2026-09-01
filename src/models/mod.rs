//! Model implementations behind one [`CausalLM`] interface.
//!
//! Each architecture lives in its own module and owns its config parsing, weight
//! layout and forward pass; everything generic over architectures — the decode
//! loop in [`crate::generation`], the KV cache, the benchmark binaries — talks to
//! [`CausalLM`] and [`ModelMeta`] instead of a concrete model type.
//!
//! | Module | Checkpoints | Distinguishing features |
//! |---|---|---|
//! | [`gpt2`] | `openai-community/gpt2{,-medium,-large,-xl}` | learned position embeddings, LayerNorm, MHA, tanh-GELU MLP, Conv1D-transposed weights |
//! | [`qwen3`] | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B}` | RoPE, RMSNorm, GQA, per-head QK-norm, SwiGLU MLP |

pub mod gpt2;
pub mod qwen3;

use candle_core::{Device, Result, Tensor};

use crate::dtype::Precision;
use crate::kv_cache::{KvCache, KvQuant};

pub use gpt2::GPT2Model;
pub use qwen3::Qwen3Model;

/// Architecture-independent facts about a loaded model.
///
/// This is what the cache, the benchmark harnesses and the memory accounting
/// need; it deliberately does not expose anything architecture-specific.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMeta {
    /// Short architecture name, e.g. `"gpt2"` or `"qwen3"`.
    pub architecture: &'static str,
    /// Number of transformer blocks — also the number of KV-cache layers.
    pub n_layer: usize,
    /// Query heads per block.
    pub n_head: usize,
    /// Key/value heads per block. Equals `n_head` for multi-head attention;
    /// smaller under grouped-query attention. **This**, not `n_head`, is what
    /// the KV cache stores per position.
    pub n_kv_head: usize,
    /// Width of one attention head. Not always `hidden_size / n_head` — Qwen3
    /// sets it independently in the checkpoint config.
    pub head_dim: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Maximum position the model can attend over.
    pub n_ctx: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Parameter count, counted from the checkpoint's own config.
    pub n_params: usize,
}

impl ModelMeta {
    /// Query heads per key/value head. `1` for plain multi-head attention.
    pub fn n_kv_groups(&self) -> usize {
        self.n_head / self.n_kv_head
    }

    /// Bytes of KV cache retained per sequence position at `bytes_per_elem`.
    ///
    /// `2` for K and V, times the layer count, times the *key/value* head count
    /// times the head width. Under GQA this is `n_kv_groups()` times smaller than
    /// the naive `n_head`-based figure.
    pub fn kv_cache_bytes_per_token(&self, bytes_per_elem: usize) -> usize {
        2 * self.n_layer * self.n_kv_head * self.head_dim * bytes_per_elem
    }

    /// [`kv_cache_bytes_per_token`](Self::kv_cache_bytes_per_token) for a
    /// floating-point precision.
    pub fn kv_cache_bytes_per_token_at(&self, precision: Precision) -> usize {
        self.kv_cache_bytes_per_token(precision.size_in_bytes())
    }

    /// Retained KV-cache bytes per position under a quantization mode.
    ///
    /// `precision` only matters for [`KvQuant::None`]; the quantized modes store
    /// fixed-width values plus fp32 scales either way.
    pub fn kv_cache_bytes_per_token_with(&self, quant: KvQuant, precision: Precision) -> usize {
        quant.bytes_per_token(
            self.n_layer,
            self.n_kv_head,
            self.head_dim,
            precision.size_in_bytes(),
        )
    }
}

/// A decoder-only language model that can run with or without a KV cache.
///
/// Object-safe on purpose: [`crate::loader`] picks an implementation from the
/// checkpoint's `config.json` at runtime and hands back a `Box<dyn CausalLM>`.
///
/// Every method takes `&self`; the mutable decode state lives in the caller's
/// [`KvCache`], so one loaded model can serve several independent sequences.
pub trait CausalLM: Send + Sync {
    /// Architecture-independent shape facts. See [`ModelMeta`].
    fn meta(&self) -> &ModelMeta;

    /// Device the weights live on.
    fn device(&self) -> &Device;

    /// Precision the weights are held at.
    fn precision(&self) -> Precision;

    /// Full-sequence forward pass, logits for **every** position:
    /// `[batch, seq, vocab]`.
    ///
    /// Recomputes attention over the whole input on every call — O(n^2) when
    /// driven as a decode loop. Kept as the reference path that
    /// [`forward_with_cache`](Self::forward_with_cache) is checked against, and
    /// used by perplexity scoring, which genuinely needs every position.
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// [`forward`](Self::forward) with the LM head applied only to the final
    /// position. Returns `[batch, vocab]`.
    ///
    /// Numerically identical to `forward(ids)` narrowed to its last position, at
    /// a fraction of the cost: the `[seq, hidden] x [hidden, vocab]` projection
    /// dominates a prefill-sized input, and every position but the last would be
    /// computed and immediately discarded.
    fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// Incremental forward pass over the **new** tokens only, reading and
    /// extending `cache`. Returns `[batch, new_seq, vocab]`.
    ///
    /// Pass the whole prompt on the first call (prefill), then one token per
    /// step. With an empty cache and the full sequence this computes exactly what
    /// [`forward`](Self::forward) does.
    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor>;

    /// [`forward_with_cache`](Self::forward_with_cache) with the LM head applied
    /// only to the final new position. Returns `[batch, vocab]`.
    fn forward_with_cache_last(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor>;

    /// A cache sized for this model, storing K/V at the model's own precision.
    fn new_cache(&self) -> KvCache {
        KvCache::new(self.meta().n_layer)
    }
}

/// Shared prologue for the cached forward paths: validate the cache against the
/// model and return `(past_len, total_len)`.
pub(crate) fn check_cache(
    cache: &KvCache,
    n_layer: usize,
    seq_len: usize,
    n_ctx: usize,
) -> Result<(usize, usize)> {
    if cache.n_layer() != n_layer {
        return Err(candle_core::Error::Msg(format!(
            "cache has {} layers, model has {n_layer}",
            cache.n_layer()
        )));
    }
    let past = cache.len();
    let total = past + seq_len;
    if total > n_ctx {
        return Err(candle_core::Error::Msg(format!(
            "sequence length {total} exceeds the model context window {n_ctx}"
        )));
    }
    Ok((past, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(n_head: usize, n_kv_head: usize, n_layer: usize, head_dim: usize) -> ModelMeta {
        ModelMeta {
            architecture: "test",
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            hidden_size: n_head * head_dim,
            n_ctx: 1024,
            vocab_size: 32,
            n_params: 0,
        }
    }

    #[test]
    fn mha_has_one_query_head_per_kv_head() {
        assert_eq!(meta(12, 12, 12, 64).n_kv_groups(), 1);
    }

    #[test]
    fn gqa_groups_divide_the_head_count() {
        // Qwen3-0.6B: 16 query heads over 8 KV heads.
        assert_eq!(meta(16, 8, 28, 128).n_kv_groups(), 2);
    }

    #[test]
    fn gpt2_kv_bytes_match_the_documented_figure() {
        // gpt2 124M at fp32: 2 * 12 layers * 12 heads * 64 dim * 4 B = 72 KiB/token.
        let m = meta(12, 12, 12, 64);
        assert_eq!(m.kv_cache_bytes_per_token(4), 73_728);
        assert_eq!(m.kv_cache_bytes_per_token_at(Precision::F32), 73_728);
    }

    #[test]
    fn gqa_shrinks_the_cache_by_the_group_count() {
        let mha = meta(16, 16, 28, 128);
        let gqa = meta(16, 8, 28, 128);
        assert_eq!(
            mha.kv_cache_bytes_per_token(2),
            gqa.kv_cache_bytes_per_token(2) * 2
        );
    }

    #[test]
    fn bf16_cache_is_half_of_fp32() {
        let m = meta(16, 8, 28, 128);
        assert_eq!(
            m.kv_cache_bytes_per_token_at(Precision::BF16) * 2,
            m.kv_cache_bytes_per_token_at(Precision::F32)
        );
    }

    #[test]
    fn check_cache_rejects_a_layer_count_mismatch() {
        let cache = KvCache::new(4);
        assert!(check_cache(&cache, 12, 1, 1024).is_err());
        assert!(check_cache(&cache, 4, 1, 1024).is_ok());
    }

    #[test]
    fn check_cache_rejects_overrunning_the_context_window() {
        let cache = KvCache::new(2);
        assert!(check_cache(&cache, 2, 5, 4).is_err());
        assert_eq!(check_cache(&cache, 2, 4, 4).unwrap(), (0, 4));
    }
}
