use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use super::attention::MultiHeadAttention;
use super::config::GPT2Config;
use crate::kv_cache::LayerKvCache;
use crate::layers::gelu;

pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp_c_fc: Linear,
    mlp_c_proj: Linear,
}

impl TransformerBlock {
    pub fn new(cfg: &GPT2Config, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::new(cfg, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;

        // Conv1D-style `[in, out]` weights, as in `attention.rs`.
        let mlp_c_fc_vb = vb.pp("mlp.c_fc");
        let mlp_c_fc_weight = mlp_c_fc_vb.get((cfg.n_embd, 4 * cfg.n_embd), "weight")?.t()?;
        let mlp_c_fc_bias = mlp_c_fc_vb.get(4 * cfg.n_embd, "bias")?;
        let mlp_c_fc = Linear::new(mlp_c_fc_weight, Some(mlp_c_fc_bias));

        let mlp_c_proj_vb = vb.pp("mlp.c_proj");
        let mlp_c_proj_weight = mlp_c_proj_vb.get((4 * cfg.n_embd, cfg.n_embd), "weight")?.t()?;
        let mlp_c_proj_bias = mlp_c_proj_vb.get(cfg.n_embd, "bias")?;
        let mlp_c_proj = Linear::new(mlp_c_proj_weight, Some(mlp_c_proj_bias));

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp_c_fc,
            mlp_c_proj,
        })
    }

    /// Position-wise feed-forward: `c_fc` -> gelu -> `c_proj`.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.mlp_c_fc)?;
        let h = gelu(&h)?;
        h.apply(&self.mlp_c_proj)
    }

    /// Full-sequence pre-LN block: attention + MLP, each with a residual.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_out = self.attn.forward(&x.apply(&self.ln_1)?, mask)?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp(&x.apply(&self.ln_2)?)?;
        x + mlp_out
    }

    /// Incremental block: the attention step reads and extends `cache`; the MLP
    /// runs on the new positions only. See
    /// [`MultiHeadAttention::forward_with_cache`].
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let attn_out = self
            .attn
            .forward_with_cache(&x.apply(&self.ln_1)?, mask, cache)?;
        let x = (x + attn_out)?;
        let mlp_out = self.mlp(&x.apply(&self.ln_2)?)?;
        x + mlp_out
    }
}
