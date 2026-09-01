//! Pointwise activations shared by the model implementations.

use candle_core::{Result, Tensor};
use std::f64::consts::PI;

/// GPT-2's `gelu_new` / `gelu_pytorch_tanh`: the tanh approximation to GELU.
///
/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// `x^3` is two multiplies rather than `powf(3.0)` (a libm `pow` call per
/// element): ~2x faster net of this function's share of a forward pass, for the
/// same result to fp32 rounding.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    let x_squared = (x * x)?;
    let x_cubed = (x_squared * x)?;
    let inner = (x + (0.044715 * x_cubed)?)?;
    let sqrt = (2.0 / PI).sqrt();
    let tanh_able = (inner * sqrt)?;
    let tanh_ed = tanh_able.tanh()?;
    let one_plus_tanh_ed = (1.0 + tanh_ed)?;

    (x * &one_plus_tanh_ed)? * 0.5
}

/// SiLU / swish: `x * sigmoid(x)`. The gate activation in Qwen3's SwiGLU MLP.
///
/// Delegates to candle's kernel, which has a fused CPU/CUDA implementation.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn vals(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn tensor(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn gelu_matches_reference_values() {
        // Reference from PyTorch nn.functional.gelu(approximate="tanh").
        let x = tensor(&[-2.0, -0.5, 0.0, 0.5, 2.0]);
        let got = vals(&gelu(&x).unwrap());
        let want = [-0.0454022, -0.154_286, 0.0, 0.345714, 1.9545977];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "gelu: got {g}, want {w}");
        }
    }

    #[test]
    fn gelu_is_near_identity_for_large_positive_x() {
        let x = tensor(&[10.0, 20.0]);
        let got = vals(&gelu(&x).unwrap());
        assert!((got[0] - 10.0).abs() < 1e-4);
        assert!((got[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn silu_matches_x_times_sigmoid() {
        let x = tensor(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
        let got = vals(&silu(&x).unwrap());
        for (g, &xv) in got.iter().zip(&[-3.0f32, -1.0, 0.0, 1.0, 3.0]) {
            let want = xv / (1.0 + (-xv).exp());
            assert!((g - want).abs() < 1e-6, "silu: got {g}, want {want}");
        }
    }
}
