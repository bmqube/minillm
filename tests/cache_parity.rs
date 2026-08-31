//! Cache-path correctness: greedy generation with the KV cache must produce the
//! exact same token ids as the plain full-recompute `forward`, and the prefill
//! logits must match element-for-element.
//!
//! Needs GPT-2 weights in `benchmarks/gpt2/` (see `benchmarks/README.md`), so
//! this is `#[ignore]` by default. Run it with:
//!
//! ```text
//! cargo test --test cache_parity -- --ignored
//! ```

use candle_core::{IndexOp, Tensor};
use minillm::kv_cache::KvCache;
use minillm::{device, loader};

const MODEL_DIR: &str = "benchmarks/gpt2";

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn cache_matches_no_cache() {
    let dev = device::best();
    let (model, tok) = loader::load(MODEL_DIR, &dev).expect("load benchmarks/gpt2");

    let prompt = "A transformer is a deep learning architecture that";
    let prompt_ids = tok.encode(prompt, true).unwrap().get_ids().to_vec();
    let steps = 32;

    // --- Reference: full recompute every step. ---
    let mut ref_seq = prompt_ids.clone();
    for _ in 0..steps {
        let input = Tensor::from_vec(ref_seq.clone(), (1, ref_seq.len()), &dev).unwrap();
        let row = model
            .forward(&input)
            .unwrap()
            .i((0, ref_seq.len() - 1))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        ref_seq.push(argmax(&row));
    }

    // --- KV cache: prefill once, then one token per step. ---
    let mut cache = KvCache::new(model.config().n_layer);
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &dev).unwrap();
    let prime = model.forward_with_cache(&input, &mut cache).unwrap();

    // Prefill logits must match the plain forward exactly (same math).
    let full = model.forward(&input).unwrap();
    let a = full.i((0, prompt_ids.len() - 1)).unwrap().to_vec1::<f32>().unwrap();
    let b = prime
        .i((0, prompt_ids.len() - 1))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-3, "prefill logits diverge: max|Δ| = {max_abs}");

    let mut cache_seq = prompt_ids.clone();
    let mut next = argmax(&b);
    cache_seq.push(next);
    for _ in 1..steps {
        let step = Tensor::from_vec(vec![next], (1, 1), &dev).unwrap();
        let row = model
            .forward_with_cache(&step, &mut cache)
            .unwrap()
            .i((0, 0))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        next = argmax(&row);
        cache_seq.push(next);
    }

    assert_eq!(
        ref_seq, cache_seq,
        "KV-cache generation diverged from the full-recompute reference"
    );
}
