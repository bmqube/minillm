use candle_core::{Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use super::config::Qwen3Config;
use crate::kv_cache::LayerKvCache;
use crate::layers::{repeat_kv, RotaryEmbedding};

/// Qwen3 attention: grouped-query, rotary, with per-head RMS normalization of Q
/// and K before the rotation.
///
/// Three differences from [`crate::models::gpt2::MultiHeadAttention`] matter:
///
/// 1. **Separate Q/K/V projections** of *different* widths — K and V are
///    `n_kv_head` wide, Q is `n_head` wide — so there is no single fused `c_attn`
///    to slice into thirds.
/// 2. **QK-norm**: an RMSNorm over each head's `head_dim` vector, applied to Q
///    and K (never V) before RoPE. Omitting it produces a model that generates
///    fluent-looking text with quietly wrong logits.
/// 3. **RoPE instead of a position embedding table**, applied at the cache
///    offset so a decode step rotates by the same angle a prefill would.
pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    n_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Qwen3Attention {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;

        // Qwen3 drops the QKV bias Qwen2 had, but honour the config rather than
        // hard-coding the assumption: a biased checkpoint would otherwise load
        // with the bias silently missing.
        let (q_proj, k_proj, v_proj) = if cfg.attention_bias {
            (
                candle_nn::linear(hidden, q_out, vb.pp("q_proj"))?,
                candle_nn::linear(hidden, kv_out, vb.pp("k_proj"))?,
                candle_nn::linear(hidden, kv_out, vb.pp("v_proj"))?,
            )
        } else {
            (
                candle_nn::linear_no_bias(hidden, q_out, vb.pp("q_proj"))?,
                candle_nn::linear_no_bias(hidden, kv_out, vb.pp("k_proj"))?,
                candle_nn::linear_no_bias(hidden, kv_out, vb.pp("v_proj"))?,
            )
        };
        // The output projection is never biased in Qwen3.
        let o_proj = candle_nn::linear_no_bias(q_out, hidden, vb.pp("o_proj"))?;

        // QK-norm is over one head's width, not the residual stream's.
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_key_value_heads,
            n_kv_groups: cfg.n_kv_groups(),
            head_dim,
            hidden_size: hidden,
        })
    }

    /// Project `x` (`[batch, seq, hidden]`) into Q, K, V.
    ///
    /// Q is `[batch, n_head, seq, head_dim]`; K and V are
    /// `[batch, n_kv_head, seq, head_dim]` — deliberately *not* expanded to the
    /// query head count, because these are what gets cached.
    ///
    /// QK-norm runs on the `[batch, seq, heads, head_dim]` layout, where the last
    /// axis is exactly one head's vector, before the transpose to head-major.
    fn project_qkv(
        &self,
        x: &Tensor,
        offset: usize,
        rope: &RotaryEmbedding,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq, _) = x.dims3()?;

        let q = x
            .apply(&self.q_proj)?
            .reshape((batch, seq, self.n_head, self.head_dim))?
            .apply(&self.q_norm)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = x
            .apply(&self.k_proj)?
            .reshape((batch, seq, self.n_kv_head, self.head_dim))?
            .apply(&self.k_norm)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = x
            .apply(&self.v_proj)?
            .reshape((batch, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Rotate Q and K only — V carries no positional information.
        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;
        Ok((q, k, v))
    }

    /// Scaled dot-product attention over cached K/V.
    ///
    /// `q` is `[batch, n_head, q_len, head_dim]`; `k`/`v` are
    /// `[batch, n_kv_head, kv_len, head_dim]` and are expanded to the query head
    /// count here, per step, never in the cache. The optional additive `mask` is
    /// `[q_len, kv_len]` and must already be in the activation dtype.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, _, q_len, head_dim) = q.dims4()?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let k = repeat_kv(k, self.n_kv_groups)?;
        let v = repeat_kv(v, self.n_kv_groups)?;

        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut scores = (scores * scale)?;

        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }

        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v.contiguous()?)?;

        out.transpose(1, 2)?
            .reshape((batch, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// Full-sequence attention with no cache, positions starting at 0.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(x, 0, rope)?;
        self.attend(&q, &k, &v, mask)
    }

    /// Incremental attention. `x` holds only the new tokens; their (narrow) K/V
    /// are appended to `cache` and the new queries attend over the full history.
    /// `past` is the cache length before this call and drives both the RoPE
    /// offset and the mask offset.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        past: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (q, k_new, v_new) = self.project_qkv(x, past, rope)?;
        let (k, v) = cache.append(&k_new, &v_new)?;
        self.attend(&q, &k, &v, mask)
    }
}
