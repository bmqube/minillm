//! Additive attention masks.

use candle_core::{DType, Device, Result, Tensor};

/// The additive penalty applied to disallowed attention positions.
///
/// Large enough to zero the softmax weight, small enough to stay finite in
/// fp16 (whose max is ~65504) — `-inf` or `-1e10` would overflow to `-inf` on
/// conversion and produce `NaN` for a row that is entirely masked.
const MASK_NEG: f32 = -3.0e4;

/// Additive causal mask, `[q_len, kv_len]`: `0` where a query position may
/// attend to a key position and a large negative value where it may not.
///
/// Query row `i` is the token at absolute position `past + i` and may attend to
/// key columns `0..=past + i`. With `past == 0` and `q_len == kv_len` this is the
/// plain lower-triangular causal mask.
///
/// `dtype` must match the attention scores the mask is broadcast onto; candle
/// will not implicitly convert.
pub fn causal(
    q_len: usize,
    kv_len: usize,
    past: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let allowed = past + i; // last key column this query may see
        for j in (allowed + 1)..kv_len {
            data[i * kv_len + j] = MASK_NEG;
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), device)?.to_dtype(dtype)
}

/// [`causal`], additionally restricted to a sliding window of `window` keys.
///
/// Query at absolute position `p` attends to keys in `(p - window, p]`. Used by
/// architectures that alternate local and global attention layers; `window == 0`
/// is treated as "no windowing" and falls back to [`causal`].
pub fn causal_sliding(
    q_len: usize,
    kv_len: usize,
    past: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if window == 0 {
        return causal(q_len, kv_len, past, dtype, device);
    }
    let mut data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let pos = past + i;
        let lowest = pos.saturating_sub(window - 1);
        for j in 0..kv_len {
            if j > pos || j < lowest {
                data[i * kv_len + j] = MASK_NEG;
            }
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    #[test]
    fn plain_causal_is_lower_triangular() {
        let m = causal(3, 3, 0, DType::F32, &Device::Cpu).unwrap();
        let r = rows(&m);
        assert_eq!(r[0], vec![0.0, MASK_NEG, MASK_NEG]);
        assert_eq!(r[1], vec![0.0, 0.0, MASK_NEG]);
        assert_eq!(r[2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn offset_mask_lets_new_queries_see_the_whole_past() {
        // 2 new queries at absolute positions 4 and 5, over 6 keys.
        let m = causal(2, 6, 4, DType::F32, &Device::Cpu).unwrap();
        let r = rows(&m);
        assert_eq!(r[0], vec![0.0, 0.0, 0.0, 0.0, 0.0, MASK_NEG]);
        assert_eq!(r[1], vec![0.0; 6]);
    }

    #[test]
    fn single_query_row_is_all_visible() {
        let m = causal(1, 8, 7, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rows(&m)[0], vec![0.0; 8]);
    }

    #[test]
    fn mask_is_finite_in_f16() {
        // -1e10 would round to -inf in f16 and poison a fully-masked softmax row.
        let m = causal(2, 2, 0, DType::F16, &Device::Cpu).unwrap();
        for v in rows(&m).into_iter().flatten() {
            assert!(v.is_finite(), "mask value {v} is not finite in f16");
        }
    }

    #[test]
    fn sliding_window_drops_keys_outside_the_window() {
        // window = 2: position 3 sees keys 2 and 3 only.
        let m = causal_sliding(1, 5, 3, 2, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rows(&m)[0], vec![MASK_NEG, MASK_NEG, 0.0, 0.0, MASK_NEG]);
    }

    #[test]
    fn sliding_window_zero_means_no_window() {
        let a = causal_sliding(3, 3, 0, 0, DType::F32, &Device::Cpu).unwrap();
        let b = causal(3, 3, 0, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rows(&a), rows(&b));
    }
}
