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

/// Scaled dot-product attention with grouped-query heads, without expanding
/// K/V.
///
/// `q` is `[batch, n_kv_head * n_rep, q_len, head_dim]`; `k`/`v` are
/// `[batch, n_kv_head, kv_len, head_dim]` and may be strided views (a KV-cache
/// buffer). The optional additive `mask` is `[q_len, kv_len]` in the activation
/// dtype. Returns `[batch, n_kv_head * n_rep, q_len, head_dim]`.
///
/// Query head `h` attends with KV head `h / n_rep` ([`repeat_kv`]'s mapping), so
/// each KV head's query group is contiguous and folds into the row axis for
/// free: `q` becomes `[batch, n_kv_head, n_rep * q_len, head_dim]` and one
/// matmul per KV head serves its whole group. Equivalent to running ordinary
/// attention on `repeat_kv(k, n_rep)` / `repeat_kv(v, n_rep)`, minus the copy of
/// the expanded K and V that would cost every decode step.
pub fn grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    n_rep: usize,
) -> Result<Tensor> {
    let (batch, n_head, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_head, kv_len, _) = k.dims4()?;
    if n_head != n_kv_head * n_rep {
        return Err(candle_core::Error::Msg(format!(
            "grouped_attention: {n_head} query heads is not {n_kv_head} KV heads x {n_rep}"
        )));
    }
    let scale = 1.0 / (head_dim as f64).sqrt();

    let q = q.reshape((batch, n_kv_head, n_rep * q_len, head_dim))?;
    // `.transpose()` alone is a free stride-swap; candle's matmul takes it as is.
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let scores = match mask {
        // Rows are (group member, query position); the mask is per position.
        Some(mask) => scores
            .reshape((batch, n_kv_head, n_rep, q_len, kv_len))?
            .broadcast_add(mask)?
            .reshape((batch, n_kv_head, n_rep * q_len, kv_len))?,
        None => scores,
    };
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    weights.matmul(v)?.reshape((batch, n_head, q_len, head_dim))
}

/// Expand grouped-query-attention keys/values so every query head has one.
///
/// `xs` is `[batch, n_kv_head, seq, head_dim]`; the result is
/// `[batch, n_kv_head * n_rep, seq, head_dim]` where output head `h` carries the
/// data of KV head `h / n_rep` — the `repeat_interleave` semantics the reference
/// implementations use.
///
/// Attention no longer calls this — [`grouped_attention`] gets the same result
/// without the copy. It stays as the reference that mapping is tested against.
/// The KV cache always stores the narrow `n_kv_head` version, which is the
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

    /// Plain attention over explicitly expanded K/V: the reference
    /// `grouped_attention` must match.
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        n_rep: usize,
    ) -> Tensor {
        let head_dim = q.dims4().unwrap().3;
        let k = repeat_kv(k, n_rep).unwrap();
        let v = repeat_kv(v, n_rep).unwrap();
        let mut s = (q.matmul(&k.t().unwrap()).unwrap() / (head_dim as f64).sqrt()).unwrap();
        if let Some(m) = mask {
            s = s.broadcast_add(m).unwrap();
        }
        candle_nn::ops::softmax_last_dim(&s)
            .unwrap()
            .matmul(&v)
            .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn grouped_attention_matches_expanded_kv() {
        let dev = Device::Cpu;
        // 2 KV heads x 3 query heads each; prefill-shaped (q_len 4) with a causal
        // mask, and decode-shaped (q_len 1) without one.
        for (q_len, kv_len, masked) in [(4, 4, true), (1, 7, false)] {
            let q = Tensor::randn(0f32, 1.0, (1, 6, q_len, 8), &dev).unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, 2, kv_len, 8), &dev).unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, 2, kv_len, 8), &dev).unwrap();
            let m = masked.then(|| {
                mask::causal(q_len, kv_len, kv_len - q_len, candle_core::DType::F32, &dev).unwrap()
            });
            let got = grouped_attention(&q, &k, &v, m.as_ref(), 3).unwrap();
            let want = reference_attention(&q, &k, &v, m.as_ref(), 3);
            assert_eq!(got.dims(), want.dims());
            let d = max_abs_diff(&got, &want);
            assert!(d < 1e-5, "q_len {q_len}: max |diff| {d}");
        }
    }

    #[test]
    fn grouped_attention_accepts_a_strided_cache_view() {
        // K/V as the KV cache hands them over: the filled prefix of a larger
        // buffer, strided along the sequence axis.
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 4, 1, 8), &dev).unwrap();
        let k_buf = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev).unwrap();
        let v_buf = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev).unwrap();
        let (k, v) = (
            k_buf.narrow(2, 0, 5).unwrap(),
            v_buf.narrow(2, 0, 5).unwrap(),
        );
        assert!(!k.is_contiguous());
        let got = grouped_attention(&q, &k, &v, None, 2).unwrap();
        let want = reference_attention(
            &q,
            &k.contiguous().unwrap(),
            &v.contiguous().unwrap(),
            None,
            2,
        );
        assert!(max_abs_diff(&got, &want) < 1e-5);
    }

    #[test]
    fn grouped_attention_rejects_mismatched_heads() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 5, 1, 4), candle_core::DType::F32, &dev).unwrap();
        let kv = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &dev).unwrap();
        assert!(grouped_attention(&q, &kv, &kv, None, 2).is_err());
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
