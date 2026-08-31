//! Cache-path correctness against the full-recompute reference.
//!
//! Needs GPT-2 weights in `benchmarks/gpt2/` (see `benchmarks/README.md`), so
//! these are `#[ignore]` by default. Run with:
//!
//! ```text
//! cargo test --test cache_parity -- --ignored
//! ```

use minillm::generation::{greedy, greedy_no_cache, Generator};
use minillm::kv_cache::KvQuant;
use minillm::{device, loader};

const MODEL_DIR: &str = "benchmarks/gpt2";
const PROMPT: &str = "A transformer is a deep learning architecture that";
const STEPS: usize = 32;

/// Greedy decoding with an fp32 KV cache must produce the exact same token ids
/// as recomputing the whole sequence every step.
#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn cache_matches_no_cache() {
    let dev = device::best();
    let (model, tok) = loader::load(MODEL_DIR, &dev).expect("load benchmarks/gpt2");
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let reference = greedy_no_cache(&model, &dev, &ids, STEPS).unwrap();
    let cached = greedy(&model, &dev, &ids, STEPS, KvQuant::None).unwrap();

    assert_eq!(
        reference, cached,
        "KV-cache generation diverged from the full-recompute reference"
    );
}

/// Quantized caches are lossy by construction, so they are not required to match
/// token-for-token — but the prefill logits must stay close, and int8 must be
/// closer than int4.
#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn quantized_caches_stay_close_to_fp32() {
    let dev = device::best();
    let (model, tok) = loader::load(MODEL_DIR, &dev).expect("load benchmarks/gpt2");
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let logits_for = |quant| {
        let mut generator = Generator::with_quant(&model, &dev, quant);
        generator
            .prefill(&ids)
            .unwrap()
            .to_vec1::<f32>()
            .expect("logits row")
    };

    let fp32 = logits_for(KvQuant::None);
    let int8 = logits_for(KvQuant::Int8);
    let int4 = logits_for(KvQuant::Int4);

    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let d8 = max_abs(&fp32, &int8);
    let d4 = max_abs(&fp32, &int4);

    assert!(
        d8 < 0.5,
        "int8 prefill logits drifted too far: max|d| = {d8}"
    );
    assert!(
        d4 < 5.0,
        "int4 prefill logits drifted too far: max|d| = {d4}"
    );
    assert!(
        d8 < d4,
        "int8 ({d8}) should be closer to fp32 than int4 ({d4})"
    );
}
