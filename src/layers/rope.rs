//! Rotary position embeddings (RoPE).
//!
//! GPT-2 learns an explicit position embedding table; every modern decoder in
//! this crate instead rotates Q and K by an angle that depends on absolute
//! position, so attention scores end up depending only on *relative* position.
//!
//! The convention here is the "half split" one used by Llama, Qwen and the rest
//! of the HuggingFace `LlamaRotaryEmbedding` lineage: the `head_dim` vector is
//! cut in half and the two halves are treated as the real and imaginary parts of
//! `head_dim / 2` complex numbers, each rotated by `pos * theta_i`. (The other
//! convention, GPT-J's, interleaves adjacent pairs instead — using the wrong one
//! against a checkpoint produces plausible-looking but wrong logits.)

use candle_core::{DType, Device, Result, Tensor};

/// Precomputed `cos` / `sin` tables for [`RotaryEmbedding::apply`].
#[derive(Debug)]
pub struct RotaryEmbedding {
    /// `[max_seq, head_dim / 2]`, in the model dtype.
    cos: Tensor,
    /// `[max_seq, head_dim / 2]`, in the model dtype.
    sin: Tensor,
    max_seq: usize,
}

impl RotaryEmbedding {
    /// Build tables covering positions `0..max_seq` for a `head_dim`-wide head.
    ///
    /// `theta` is the base (`rope_theta` in a HuggingFace config; 10000 for the
    /// original formulation, 1e6 for Qwen3). Angles are computed in fp32 and cast
    /// to `dtype` at the end — the same order the reference implementations use,
    /// so a bf16 model gets bit-comparable tables.
    pub fn new(
        head_dim: usize,
        max_seq: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(candle_core::Error::Msg(format!(
                "RoPE needs an even head_dim, got {head_dim}"
            )));
        }
        let half = head_dim / 2;

        // inv_freq[i] = theta ^ (-2i / head_dim), i in 0..half
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (theta.powf(-(2.0 * i as f64) / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;

        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq, 1), device)?;

        // [max_seq, half] outer product of position and inverse frequency.
        let freqs = positions.broadcast_mul(&inv_freq)?;

        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            max_seq,
        })
    }

    /// Highest position these tables cover, exclusive.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Rotate `x` (`[batch, n_head, seq, head_dim]`) in place of positions
    /// `offset..offset + seq`.
    ///
    /// `offset` is the KV-cache length, so a decode step rotating its single new
    /// token gets the same angle it would have had in a full-sequence prefill.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq, _) = x.dims4()?;
        if offset + seq > self.max_seq {
            return Err(candle_core::Error::Msg(format!(
                "RoPE position {} exceeds the precomputed table ({} entries)",
                offset + seq,
                self.max_seq
            )));
        }
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference: rotate the half-split pairs of one `head_dim` vector.
    fn reference(x: &[f32], pos: usize, theta: f64) -> Vec<f32> {
        let d = x.len();
        let half = d / 2;
        let mut out = vec![0.0f32; d];
        for i in 0..half {
            let freq = theta.powf(-(2.0 * i as f64) / d as f64);
            let angle = (pos as f64) * freq;
            let (c, s) = (angle.cos() as f32, angle.sin() as f32);
            out[i] = x[i] * c - x[i + half] * s;
            out[i + half] = x[i + half] * c + x[i] * s;
        }
        out
    }

    fn rotate_one(x: &[f32], pos: usize, theta: f64) -> Vec<f32> {
        let d = x.len();
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(d, 64, theta, DType::F32, &dev).unwrap();
        let t = Tensor::from_vec(x.to_vec(), (1, 1, 1, d), &dev).unwrap();
        rope.apply(&t, pos)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn matches_the_scalar_reference() {
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 - 1.0).collect();
        for pos in [0usize, 1, 5, 33] {
            let got = rotate_one(&x, pos, 10000.0);
            let want = reference(&x, pos, 10000.0);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "pos {pos}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn position_zero_is_the_identity() {
        let x: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let got = rotate_one(&x, 0, 10000.0);
        for (g, w) in got.iter().zip(&x) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn rotation_preserves_pairwise_norm() {
        // Each (i, i+half) pair is rotated, so its 2-norm must be unchanged.
        let x: Vec<f32> = vec![0.3, -1.2, 2.0, 0.7];
        let got = rotate_one(&x, 11, 1_000_000.0);
        let before = (x[0] * x[0] + x[2] * x[2]).sqrt();
        let after = (got[0] * got[0] + got[2] * got[2]).sqrt();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn offset_decode_matches_full_prefill() {
        // The whole point of `offset`: rotating token 3 alone must equal the
        // fourth row of rotating a length-4 sequence in one pass.
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 16, 10000.0, DType::F32, &dev).unwrap();
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.11).collect();

        let full = Tensor::from_vec(data.clone(), (1, 1, 4, 4), &dev).unwrap();
        let full_out = rope.apply(&full, 0).unwrap();
        let last_row = full_out.narrow(2, 3, 1).unwrap();

        let step = Tensor::from_vec(data[12..].to_vec(), (1, 1, 1, 4), &dev).unwrap();
        let step_out = rope.apply(&step, 3).unwrap();

        let a = last_row.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = step_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn rejects_odd_head_dim() {
        assert!(RotaryEmbedding::new(5, 8, 10000.0, DType::F32, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_positions_past_the_table() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 8, 10000.0, DType::F32, &dev).unwrap();
        let t = Tensor::zeros((1, 1, 2, 4), DType::F32, &dev).unwrap();
        assert!(rope.apply(&t, 7).is_err());
        assert!(rope.apply(&t, 6).is_ok());
    }
}
