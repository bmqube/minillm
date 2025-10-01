use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::GPT2Config;

pub struct MultiHeadAttention {
    c_attn: Linear, // Combined Q, K, V projection
    c_proj: Linear, // Output projection
    n_head: usize,
    n_embd: usize,
}

impl MultiHeadAttention {
    pub fn new(cfg: &GPT2Config, vb: VarBuilder) -> Result<Self> {
        // let c_attn = candle_nn::linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
        // let c_proj = candle_nn::linear(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        let c_attn_weight = vb.get((cfg.n_embd, 3 * cfg.n_embd), "c_attn.weight")?.t()?;
        let c_attn_bias = vb.get(3 * cfg.n_embd, "c_attn.bias")?;
        let c_attn = candle_nn::Linear::new(c_attn_weight, Some(c_attn_bias));

        // Manually load and transpose c_proj weights for GPT-2 compatibility
        let c_proj_vb = vb.pp("c_proj");
        let c_proj_weight = c_proj_vb.get((cfg.n_embd, cfg.n_embd), "weight")?.t()?;
        let c_proj_bias = c_proj_vb.get(cfg.n_embd, "bias")?;
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));

        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Combined QKV projection
        let qkv = x.apply(&self.c_attn)?;
        let qkv = qkv.reshape((
            batch_size,
            seq_len,
            3,
            self.n_head,
            self.n_embd / self.n_head,
        ))?;

        // Split and transpose for attention
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?.contiguous()?; // [batch, heads, seq, head_dim]
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let head_dim = self.n_embd / self.n_head;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut scores = (scores * scale)?;

        // Apply causal mask
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn_weights.matmul(&v)?;

        // Concatenate heads and project
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.n_embd))?;
        out.apply(&self.c_proj)
    }
}
