use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Qwen3.5's RMSNorm: `x̂ · (1 + w)`, with the gain stored **zero-centred**.
///
/// Every RMSNorm in the model (block norms, final norm, QK-norm) uses it. A
/// checkpoint's `weight` is near 0, not near 1, so loading it into a plain
/// `x̂ · w` RMSNorm — the Qwen3 one — scales every activation towards zero, and
/// the model still produces fluent-looking text.
///
/// The reference computes in fp32 and casts the result back, so `1 + w` is
/// formed once at load **in fp32**: in bf16, `1 + 0.0123` rounds to `1.0156`.
pub struct ZeroCenteredRmsNorm {
    /// `1 + w`, fp32.
    gain: Tensor,
    eps: f32,
}

impl ZeroCenteredRmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?.to_dtype(DType::F32)?;
        Ok(Self {
            gain: (w + 1.0)?,
            eps: eps as f32,
        })
    }
}

impl Module for ZeroCenteredRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        candle_nn::ops::rms_norm(&x, &self.gain, self.eps)?.to_dtype(dtype)
    }
}

/// The DeltaNet output norm: `w · x̂`, then multiplied by `silu(gate)`.
///
/// Unlike [`ZeroCenteredRmsNorm`] the gain here is a **plain** one, stored
/// near one. Normalisation runs in fp32 and is cast back before the gain, as
/// the reference does.
pub struct GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GatedRmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    /// `x` and `gate` share a shape whose last axis is the normalised width.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let inv_rms = (xf.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        let normed = xf.broadcast_mul(&inv_rms)?.to_dtype(dtype)?;
        let scaled = normed.broadcast_mul(&self.weight.to_dtype(dtype)?)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        (scaled.to_dtype(DType::F32)? * gate)?.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    fn vb_with(name: &str, values: Vec<f32>) -> VarBuilder<'static> {
        let n = values.len();
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            Tensor::from_vec(values, n, &Device::Cpu).unwrap(),
        );
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn zero_weight_is_a_unit_gain() {
        // w = 0 must behave like a plain RMSNorm with gain 1.
        let norm = ZeroCenteredRmsNorm::new(4, 1e-6, vb_with("weight", vec![0.0; 4])).unwrap();
        let x = Tensor::new(&[[3.0f32, -4.0, 0.0, 0.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        // rms = sqrt(25 / 4) = 2.5
        let want = [1.2f32, -1.6, 0.0, 0.0];
        for (g, w) in y[0].iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
    }

    #[test]
    fn gated_norm_multiplies_by_silu_of_the_gate() {
        let norm = GatedRmsNorm::new(2, 1e-6, vb_with("weight", vec![2.0, 2.0])).unwrap();
        let x = Tensor::new(&[[1.0f32, 1.0]], &Device::Cpu).unwrap();
        let gate = Tensor::new(&[[0.0f32, 10.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x, &gate).unwrap().to_vec2::<f32>().unwrap();
        // x̂ = [1, 1]; gain 2; silu(0) = 0, silu(10) ≈ 9.99955
        assert!(y[0][0].abs() < 1e-6);
        assert!((y[0][1] - 2.0 * 9.999_546).abs() < 1e-3);
    }
}
