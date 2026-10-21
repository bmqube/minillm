use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen35Config;
use super::norm::GatedRmsNorm;
use crate::kv_cache::RecurrentCache;

/// Gated DeltaNet: the linear-attention mixer in three of every four Qwen3.5
/// layers.
///
/// Instead of attending over a growing history, each value head keeps a
/// `[key_dim, value_dim]` matrix `S` — a fast-weight memory mapping keys to
/// values — and updates it once per token:
///
/// ```text
/// S  ← S · exp(g)                     decay  (g ≤ 0, learned per head and token)
/// S  ← S + β · k ⊗ (v − Sᵀk)          delta rule: write only what S gets wrong
/// o  = Sᵀq
/// ```
///
/// with `k`, `q` L2-normalised and `q` scaled by `1/√key_dim`. Before that, Q/K/V
/// pass through a short causal depthwise convolution (kernel 4) and SiLU; after
/// it, a [`GatedRmsNorm`] gated by a separate projection `z`.
///
/// Decode state is constant-size: `S` per head plus the convolution's last
/// `kernel - 1` inputs, held in a [`RecurrentCache`]. Both are kept in fp32, as
/// the reference does (`mamba_ssm_dtype: float32`).
///
/// The update runs token by token for prefill as well as decode — correct, and
/// one code path, but O(seq) small ops per layer. The reference prefills with a
/// chunked formulation that is equal in exact arithmetic and much faster on long
/// prompts; that is an optimisation for later.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    /// Depthwise kernel, `[conv_dim, kernel]`.
    conv_weight: Tensor,
    /// `-exp(A_log)` per value head, fp32.
    neg_a: Tensor,
    /// fp32, per value head.
    dt_bias: Tensor,
    norm: GatedRmsNorm,
    out_proj: Linear,
    n_k_heads: usize,
    n_v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    kernel: usize,
}

impl GatedDeltaNet {
    pub fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nv = cfg.linear_num_value_heads;
        let value_dim = nv * cfg.linear_value_head_dim;
        let conv_dim = cfg.conv_dim();
        let kernel = cfg.linear_conv_kernel_dim;

        let conv_weight = vb
            .get((conv_dim, 1, kernel), "conv1d.weight")?
            .reshape((conv_dim, kernel))?;
        let neg_a = vb.get(nv, "A_log")?.to_dtype(DType::F32)?.exp()?.neg()?;
        let dt_bias = vb.get(nv, "dt_bias")?.to_dtype(DType::F32)?;

        Ok(Self {
            in_proj_qkv: candle_nn::linear_no_bias(h, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: candle_nn::linear_no_bias(h, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: candle_nn::linear_no_bias(h, nv, vb.pp("in_proj_b"))?,
            in_proj_a: candle_nn::linear_no_bias(h, nv, vb.pp("in_proj_a"))?,
            conv_weight,
            neg_a,
            dt_bias,
            norm: GatedRmsNorm::new(cfg.linear_value_head_dim, cfg.rms_norm_eps, vb.pp("norm"))?,
            out_proj: candle_nn::linear_no_bias(value_dim, h, vb.pp("out_proj"))?,
            n_k_heads: cfg.linear_num_key_heads,
            n_v_heads: nv,
            k_dim: cfg.linear_key_head_dim,
            v_dim: cfg.linear_value_head_dim,
            kernel,
        })
    }

    /// Causal depthwise convolution of `x` (`[b, channels, seq]`), continuing
    /// from the trailing inputs in `state` (`[b, channels, kernel - 1]`, zeros
    /// on the first call). Returns the activated output and the new state.
    fn conv(&self, x: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, c, s) = x.dims3()?;
        let tail = self.kernel - 1;
        let prev = match state {
            Some(t) => t.to_dtype(x.dtype())?,
            None => Tensor::zeros((b, c, tail), x.dtype(), x.device())?,
        };
        let full = Tensor::cat(&[&prev, x], 2)?; // [b, c, tail + s]
        let w = self.conv_weight.to_dtype(x.dtype())?;

        // out[t] = sum_j w[j] * full[t + j]: `kernel` shifted multiply-adds.
        let mut out = full
            .narrow(2, 0, s)?
            .broadcast_mul(&w.narrow(1, 0, 1)?.unsqueeze(0)?)?;
        for j in 1..self.kernel {
            let tap = w.narrow(1, j, 1)?.unsqueeze(0)?; // [1, c, 1]
            out = (out + full.narrow(2, j, s)?.broadcast_mul(&tap)?)?;
        }
        let new_state = full
            .narrow(2, s, tail)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        Ok((candle_nn::ops::silu(&out)?, new_state))
    }

    /// `[b, s, heads, d]` -> L2-normalised over `d`, as the reference's
    /// `l2norm(x, eps = 1e-6)`.
    fn l2norm(x: &Tensor) -> Result<Tensor> {
        let inv = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv)
    }

    /// Process the new tokens in `x` (`[b, seq, hidden]`), continuing from and
    /// updating `cache`.
    pub fn forward(&self, x: &Tensor, cache: &mut RecurrentCache) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let dtype = x.dtype();
        let (nk, nv, dk, dv) = (self.n_k_heads, self.n_v_heads, self.k_dim, self.v_dim);
        let key_dim = nk * dk;

        // Q/K/V through the causal conv, channels-first.
        let mixed = x.apply(&self.in_proj_qkv)?.transpose(1, 2)?.contiguous()?;
        let (mixed, conv_state) = self.conv(&mixed, cache.conv_state())?;
        let mixed = mixed.transpose(1, 2)?; // [b, s, conv_dim]

        let mut q = mixed.narrow(2, 0, key_dim)?.reshape((b, s, nk, dk))?;
        let mut k = mixed.narrow(2, key_dim, key_dim)?.reshape((b, s, nk, dk))?;
        let v = mixed
            .narrow(2, 2 * key_dim, nv * dv)?
            .reshape((b, s, nv, dv))?;
        if nv > nk {
            // Value head h reads key head h / rep (repeat_interleave).
            let rep = nv / nk;
            let expand = |t: Tensor| -> Result<Tensor> {
                t.unsqueeze(3)?
                    .broadcast_as((b, s, nk, rep, dk))?
                    .reshape((b, s, nv, dk))
            };
            q = expand(q)?;
            k = expand(k)?;
        }

        let z = x.apply(&self.in_proj_z)?.reshape((b, s, nv, dv))?;
        let beta = candle_nn::ops::sigmoid(&x.apply(&self.in_proj_b)?)?.to_dtype(DType::F32)?;
        let a = x.apply(&self.in_proj_a)?.to_dtype(DType::F32)?;
        let decay = softplus(&a.broadcast_add(&self.dt_bias)?)?
            .broadcast_mul(&self.neg_a)?
            .exp()?; // [b, s, nv], in (0, 1]

        // Heads-first fp32 for the recurrence.
        let to_heads = |t: &Tensor| t.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous();
        let q = (Self::l2norm(&to_heads(&q)?)? / (dk as f64).sqrt())?; // [b, nv, s, dk]
        let k = Self::l2norm(&to_heads(&k)?)?;
        let v = to_heads(&v)?; // [b, nv, s, dv]
        let decay = decay.transpose(1, 2)?.contiguous()?; // [b, nv, s]
        let beta = beta.transpose(1, 2)?.contiguous()?;

        let mut state = match cache.recurrent_state() {
            Some(t) => t.clone(),
            None => Tensor::zeros((b, nv, dk, dv), DType::F32, x.device())?,
        };
        let mut outs = Vec::with_capacity(s);
        for t in 0..s {
            let q_t = q.narrow(2, t, 1)?; // [b, nv, 1, dk]
            let k_t = k.narrow(2, t, 1)?;
            let v_t = v.narrow(2, t, 1)?; // [b, nv, 1, dv]
            let g_t = decay.narrow(2, t, 1)?.unsqueeze(3)?; // [b, nv, 1, 1]
            let beta_t = beta.narrow(2, t, 1)?.unsqueeze(3)?;

            state = state.broadcast_mul(&g_t)?;
            let recalled = k_t.matmul(&state)?; // Sᵀk: [b, nv, 1, dv]
            let delta = (v_t - recalled)?.broadcast_mul(&beta_t)?;
            state = (state + k_t.transpose(2, 3)?.matmul(&delta)?)?; // + k ⊗ delta
            outs.push(q_t.matmul(&state)?); // Sᵀq
        }
        let out = Tensor::cat(&outs, 2)?.transpose(1, 2)?.to_dtype(dtype)?; // [b, s, nv, dv]
        cache.update(conv_state, state, s);

        self.norm
            .forward(&out.contiguous()?, &z)?
            .reshape((b, s, nv * dv))?
            .apply(&self.out_proj)
    }
}

/// `log(1 + exp(x))`, stable for large `|x|`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // max(x, 0) + log(1 + exp(-|x|))
    let pos = x.relu()?;
    let tail = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    pos + tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn softplus_matches_the_definition() {
        let x = Tensor::new(&[-30.0f32, -1.0, 0.0, 2.0, 30.0], &Device::Cpu).unwrap();
        let got = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        for (g, xi) in got.iter().zip([-30.0f32, -1.0, 0.0, 2.0, 30.0]) {
            let want = if xi > 20.0 { xi } else { (1.0 + xi.exp()).ln() };
            assert!((g - want).abs() < 1e-5, "softplus({xi}) = {g}, want {want}");
        }
    }
}
