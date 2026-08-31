use candle_core::{Result, Tensor};
use std::f64::consts::PI;

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    //
    // x³ via two multiplies rather than `powf(3.0)` (a libm `pow` call per
    // element): ~2x faster net of this function's share of a forward pass,
    // for the same result to fp32 rounding.
    let x_squared = (x * x)?;
    let x_cubed = (x_squared * x)?;
    let inner = (x + (0.044715 * x_cubed)?)?;
    let sqrt = (2.0 / PI).sqrt();
    let tanh_able = (inner * sqrt)?;
    let tanh_ed = tanh_able.tanh()?;
    let one_plus_tanh_ed = (1.0 + tanh_ed)?;

    (x * &one_plus_tanh_ed)? * 0.5
}
