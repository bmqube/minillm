//! Building blocks shared by the model implementations in [`crate::models`].
//!
//! Anything a second architecture would otherwise copy lives here: activations,
//! attention masks, rotary embeddings, and the grouped-query-attention head
//! expansion. Architecture-specific wiring stays in the model module.

pub mod activation;
pub mod mask;
pub mod rope;

use candle_core::{Result, Tensor};

pub use activation::{gelu, silu};
pub use rope::RotaryEmbedding;

/// Expand grouped-query-attention keys/values so every query head has one.
///
/// `xs` is `[batch, n_kv_head, seq, head_dim]`; the result is
/// `[batch, n_kv_head * n_rep, seq, head_dim]` where output head `h` carries the
/// data of KV head `h / n_rep` — the `repeat_interleave` semantics the reference
/// implementations use.
///
/// Only the *expanded* copy is ever materialised, and only for the attention
/// matmul; the KV cache stores the narrow `n_kv_head` version, which is the
/// whole memory point of GQA. Expanding before caching would silently multiply
/// the cache size by `n_rep` (8x on Qwen3-8B).
pub fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (batch, n_kv_head, seq, head_dim) = xs.dims4()?;
    // `cat` then `reshape` is faster than a broadcast here: it avoids leaving the
    // tensor strided, which a later matmul would have to copy anyway.
    Tensor::cat(&vec![xs; n_rep], 2)?.reshape((batch, n_kv_head * n_rep, seq, head_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    /// `[1, n_kv_head, seq, 2]` where every element of KV head `h` equals `h`.
    fn kv(n_kv_head: usize, seq: usize) -> Tensor {
        let mut data = Vec::new();
        for h in 0..n_kv_head {
            for _ in 0..seq * 2 {
                data.push(h as f32);
            }
        }
        Tensor::from_vec(data, (1, n_kv_head, seq, 2), &Device::Cpu).unwrap()
    }

    #[test]
    fn n_rep_one_is_a_no_op() {
        let x = kv(4, 3);
        let out = repeat_kv(&x, 1).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn expands_head_count() {
        let out = repeat_kv(&kv(2, 5), 4).unwrap();
        assert_eq!(out.dims(), &[1, 8, 5, 2]);
    }

    #[test]
    fn output_head_h_carries_kv_head_h_over_n_rep() {
        // 2 KV heads, 4 query heads per KV head -> heads 0..3 are KV head 0,
        // heads 4..7 are KV head 1. Getting this backwards (h % n_kv_head) is
        // the classic GQA bug: it still runs and still trains, it just decodes
        // the wrong thing.
        let out = repeat_kv(&kv(2, 3), 4).unwrap();
        for h in 0..8 {
            let want = (h / 4) as f32;
            let got = out.i((0, h, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
            assert_eq!(got, want, "head {h} should come from KV head {want}");
        }
    }

    #[test]
    fn preserves_sequence_order() {
        let dev = Device::Cpu;
        // one KV head, seq 3, head_dim 1: values 10, 20, 30
        let x = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (1, 1, 3, 1), &dev).unwrap();
        let out = repeat_kv(&x, 2).unwrap();
        for h in 0..2 {
            let row: Vec<f32> = (0..3)
                .map(|t| out.i((0, h, t, 0)).unwrap().to_scalar::<f32>().unwrap())
                .collect();
            assert_eq!(row, vec![10.0, 20.0, 30.0]);
        }
    }
}
