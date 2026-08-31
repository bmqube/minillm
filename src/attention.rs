use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::GPT2Config;
use crate::kv_cache::LayerKvCache;

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

    /// Project `x` (`[batch, seq, n_embd]`) into per-head Q, K, V, each shaped
    /// `[batch, n_head, seq, head_dim]`.
    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let head_dim = self.n_embd / self.n_head;

        let qkv = x.apply(&self.c_attn)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.n_head, head_dim))?;

        // [batch, heads, seq, head_dim]
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?.contiguous()?;
        Ok((q, k, v))
    }

    /// Scaled dot-product attention. `q` is `[batch, n_head, q_len, head_dim]`,
    /// `k`/`v` are `[batch, n_head, kv_len, head_dim]`, and the optional additive
    /// `mask` is `[q_len, kv_len]`. Returns `[batch, q_len, n_embd]` after the
    /// output projection.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, _, q_len, head_dim) = q.dims4()?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // `.transpose()` alone is a free stride-swap (candle's CPU matmul takes
        // strided layouts directly, the same way it does for the `.t()`-loaded
        // Linear weights elsewhere in this crate); the `.contiguous()` that used
        // to follow it forced a full copy of `k` on every attend() call.
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let mut scores = (scores * scale)?;

        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn_weights.matmul(v)?;

        // Concatenate heads and project.
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, q_len, self.n_embd))?;
        out.apply(&self.c_proj)
    }

    /// Full-sequence attention: every position attends over the whole input.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(x)?;
        self.attend(&q, &k, &v, mask)
    }

    /// Incremental attention. `x` holds only the new tokens
    /// (`[batch, new_seq, n_embd]`); their K/V are appended to `cache` and the
    /// new queries attend over the full cached history. `mask` is
    /// `[new_seq, past + new_seq]`, or `None` for a single-token decode step
    /// (the lone new query may attend to everything).
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let (q, k_new, v_new) = self.project_qkv(x)?;
        let (k, v) = cache.append(&k_new, &v_new)?;
        self.attend(&q, &k, &v, mask)
    }
}
