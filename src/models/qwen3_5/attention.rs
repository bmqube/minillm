use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen35Config;
use super::norm::ZeroCenteredRmsNorm;
use crate::kv_cache::LayerKvCache;
use crate::layers::{grouped_attention, RotaryEmbedding};

/// Qwen3.5's full-attention layer (one in every four).
///
/// Qwen3 attention — GQA, per-head QK-norm, RoPE — with three changes:
///
/// 1. **Output gate.** `q_proj` is twice as wide as the heads need: per head it
///    emits the query *and* a gate, `[q (head_dim) | gate (head_dim)]`,
///    interleaved head by head rather than all queries then all gates. The
///    attention output is multiplied by `sigmoid(gate)` before `o_proj`.
/// 2. **Partial RoPE.** Only the first `rotary_dim` channels of each head
///    rotate (64 of 256); the rest pass through.
/// 3. **Zero-centred QK-norm** — see [`ZeroCenteredRmsNorm`].
pub struct Qwen35Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: ZeroCenteredRmsNorm,
    k_norm: ZeroCenteredRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_dim: usize,
    gated: bool,
}

impl Qwen35Attention {
    pub fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_out = cfg.num_attention_heads * hd * if cfg.attn_output_gate { 2 } else { 1 };
        let kv_out = cfg.num_key_value_heads * hd;
        let linear = |i, o, name| {
            if cfg.attention_bias {
                candle_nn::linear(i, o, vb.pp(name))
            } else {
                candle_nn::linear_no_bias(i, o, vb.pp(name))
            }
        };
        Ok(Self {
            q_proj: linear(h, q_out, "q_proj")?,
            k_proj: linear(h, kv_out, "k_proj")?,
            v_proj: linear(h, kv_out, "v_proj")?,
            o_proj: linear(cfg.num_attention_heads * hd, h, "o_proj")?,
            q_norm: ZeroCenteredRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: ZeroCenteredRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_key_value_heads,
            head_dim: hd,
            rotary_dim: cfg.rotary_dim(),
            gated: cfg.attn_output_gate,
        })
    }

    /// Rotate the first `rotary_dim` channels of `x` (`[b, heads, seq, head_dim]`)
    /// at positions starting from `offset`; pass the rest through.
    fn partial_rope(&self, x: &Tensor, offset: usize, rope: &RotaryEmbedding) -> Result<Tensor> {
        if self.rotary_dim == self.head_dim {
            return rope.apply(x, offset);
        }
        let rotated = rope.apply(&x.narrow(3, 0, self.rotary_dim)?, offset)?;
        let passed = x.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)?;
        Tensor::cat(&[&rotated, &passed], 3)
    }

    /// Attention over the new tokens in `x` (`[b, seq, hidden]`), appending
    /// their K/V to `cache`. `past` is the cache length before this call.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        past: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);

        let q_full = x.apply(&self.q_proj)?;
        let (q, gate) = if self.gated {
            let q_full = q_full.reshape((b, s, nh, 2 * hd))?;
            (q_full.narrow(3, 0, hd)?, Some(q_full.narrow(3, hd, hd)?))
        } else {
            (q_full.reshape((b, s, nh, hd))?, None)
        };

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = x
            .apply(&self.k_proj)?
            .reshape((b, s, nkv, hd))?
            .apply(&self.k_norm)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = x
            .apply(&self.v_proj)?
            .reshape((b, s, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.partial_rope(&q, past, rope)?;
        let k = self.partial_rope(&k, past, rope)?;

        let (k, v) = cache.append(&k, &v)?;
        let out = grouped_attention(&q, &k, &v, mask, nh / nkv)?
            .transpose(1, 2)?
            .reshape((b, s, nh * hd))?;
        let out = match gate {
            Some(gate) => (out * candle_nn::ops::sigmoid(&gate.reshape((b, s, nh * hd))?)?)?,
            None => out,
        };
        out.apply(&self.o_proj)
    }
}
