use crate::config::GPT2Config;
use crate::{activations::gelu, attention::MultiHeadAttention};
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp_c_fc: Linear,
    mlp_c_proj: Linear,
}

impl TransformerBlock {
    pub fn new(cfg: &GPT2Config, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::new(cfg, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_2"))?;

        // Manually load and transpose MLP weights for GPT-2 compatibility
        let mlp_c_fc_vb = vb.pp("mlp.c_fc");
        let mlp_c_fc_weight = mlp_c_fc_vb
            .get((cfg.n_embd, 4 * cfg.n_embd), "weight")?
            .t()?;
        let mlp_c_fc_bias = mlp_c_fc_vb.get(4 * cfg.n_embd, "bias")?;
        let mlp_c_fc = Linear::new(mlp_c_fc_weight, Some(mlp_c_fc_bias));

        let mlp_c_proj_vb = vb.pp("mlp.c_proj");
        let mlp_c_proj_weight = mlp_c_proj_vb
            .get((4 * cfg.n_embd, cfg.n_embd), "weight")?
            .t()?;
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

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention with residual connection
        let ln1_out = x.apply(&self.ln_1)?;
        let attn_out = self.attn.forward(&ln1_out, mask)?;
        let x = (x + attn_out)?;

        // MLP with residual connection
        let ln2_out = x.apply(&self.ln_2)?;
        let mlp_out = ln2_out.apply(&self.mlp_c_fc)?;
        let mlp_out = gelu(&mlp_out)?;
        let mlp_out = mlp_out.apply(&self.mlp_c_proj)?;

        x + mlp_out
    }
}
