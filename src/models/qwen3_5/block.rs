use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::attention::Qwen35Attention;
use super::config::Qwen35Config;
use super::delta_net::GatedDeltaNet;
use super::norm::ZeroCenteredRmsNorm;
use crate::kv_cache::KvCache;
use crate::layers::{silu, RotaryEmbedding};

/// The token mixer: what distinguishes one Qwen3.5 layer from another.
enum Mixer {
    Attention(Qwen35Attention),
    DeltaNet(GatedDeltaNet),
}

/// One Qwen3.5 decoder layer: pre-norm token mixer and pre-norm SwiGLU MLP,
/// each with a residual — Qwen3's block shape, with either gated attention or
/// a Gated DeltaNet as the mixer and zero-centred RMSNorms.
pub struct Qwen35Block {
    input_layernorm: ZeroCenteredRmsNorm,
    mixer: Mixer,
    post_attention_layernorm: ZeroCenteredRmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen35Block {
    pub fn new(cfg: &Qwen35Config, layer: usize, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let mixer = if cfg.layer_types[layer] == "full_attention" {
            Mixer::Attention(Qwen35Attention::new(cfg, vb.pp("self_attn"))?)
        } else {
            Mixer::DeltaNet(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
        };
        let mlp = vb.pp("mlp");
        Ok(Self {
            input_layernorm: ZeroCenteredRmsNorm::new(
                h,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            mixer,
            post_attention_layernorm: ZeroCenteredRmsNorm::new(
                h,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate_proj: candle_nn::linear_no_bias(h, i, mlp.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(h, i, mlp.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(i, h, mlp.pp("down_proj"))?,
        })
    }

    /// Process the new tokens in `x`, reading and extending layer `layer` of
    /// `cache`. `mask` is only consulted by attention layers; `past` is the
    /// number of positions already processed.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
        layer: usize,
        past: usize,
        rope: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let mixed = match &self.mixer {
            Mixer::Attention(attn) => {
                attn.forward(&normed, mask, cache.layer(layer), past, rope)?
            }
            Mixer::DeltaNet(net) => net.forward(&normed, cache.recurrent(layer))?,
        };
        let x = (x + mixed)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let mlp = (silu(&h.apply(&self.gate_proj)?)? * h.apply(&self.up_proj)?)?
            .apply(&self.down_proj)?;
        x + mlp
    }
}
