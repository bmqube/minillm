use candle_core::{Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use super::attention::Qwen3Attention;
use super::config::Qwen3Config;
use crate::kv_cache::LayerKvCache;
use crate::layers::{silu, RotaryEmbedding};

/// One Qwen3 decoder layer: pre-norm attention and pre-norm SwiGLU MLP, each
/// with a residual.
///
/// Structurally the same shape as GPT-2's block, with three substitutions:
/// RMSNorm for LayerNorm, SwiGLU for the 4x GELU MLP, and rotary attention for
/// the position-embedding-table kind.
pub struct Qwen3Block {
    input_layernorm: RmsNorm,
    attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Block {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        if cfg.hidden_act != "silu" {
            return Err(candle_core::Error::Msg(format!(
                "unsupported hidden_act {:?}: this crate implements only silu (SwiGLU)",
                cfg.hidden_act
            )));
        }

        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let attn = Qwen3Attention::new(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp = vb.pp("mlp");
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = candle_nn::linear_no_bias(h, i, mlp.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(h, i, mlp.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(i, h, mlp.pp("down_proj"))?;

        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))`.
    ///
    /// Two parallel projections rather than GPT-2's single `c_fc`, which is why
    /// the MLP holds three matrices instead of two for a comparable width.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&x.apply(&self.gate_proj)?)?;
        let up = x.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }

    /// Full-sequence block, positions starting at 0.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let attn_out = self
            .attn
            .forward(&x.apply(&self.input_layernorm)?, mask, rope)?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp(&x.apply(&self.post_attention_layernorm)?)?;
        x + mlp_out
    }

    /// Incremental block: attention reads and extends `cache`; the MLP runs on
    /// the new positions only.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        past: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let attn_out = self.attn.forward_with_cache(
            &x.apply(&self.input_layernorm)?,
            mask,
            cache,
            past,
            rope,
        )?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp(&x.apply(&self.post_attention_layernorm)?)?;
        x + mlp_out
    }
}
