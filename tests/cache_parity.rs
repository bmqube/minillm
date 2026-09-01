//! Cache-path correctness against the full-recompute reference.
//!
//! Needs weights on disk (see `benchmarks/README.md`), so these are `#[ignore]`
//! by default. Run with:
//!
//! ```text
//! cargo test --test cache_parity -- --ignored
//! ```
//!
//! The GPT-2 cases need `benchmarks/gpt2/`; the Qwen3 cases need
//! `benchmarks/qwen3-0.6b/`. Each case skips with a message rather than failing
//! if its checkpoint is absent, so running with one of the two present is fine.

use minillm::generation::{greedy, greedy_no_cache, Generator};
use minillm::kv_cache::KvQuant;
use minillm::models::CausalLM;
use minillm::{device, loader};

const GPT2_DIR: &str = "benchmarks/gpt2";
const QWEN3_DIR: &str = "benchmarks/qwen3-0.6b";
const PROMPT: &str = "A transformer is a deep learning architecture that";
const STEPS: usize = 32;

/// Load a checkpoint, or return `None` (with a note) if it is not on disk.
fn load(dir: &str) -> Option<(Box<dyn CausalLM>, tokenizers::Tokenizer)> {
    if !std::path::Path::new(dir).is_dir() {
        eprintln!("skipping: {dir} not present");
        return None;
    }
    Some(loader::load(dir, &device::best()).unwrap_or_else(|e| panic!("load {dir}: {e}")))
}

/// Greedy decoding with a KV cache must produce the exact same token ids as
/// recomputing the whole sequence every step.
///
/// This is the cache's own correctness gate, independent of any external
/// reference: whatever the model computes, the two paths must agree exactly.
fn cache_matches_no_cache(dir: &str) {
    let Some((model, tok)) = load(dir) else {
        return;
    };
    let model = model.as_ref();
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let reference = greedy_no_cache(model, &ids, STEPS).unwrap();
    let cached = greedy(model, &ids, STEPS, KvQuant::None).unwrap();

    assert_eq!(
        reference, cached,
        "{dir}: KV-cache generation diverged from the full-recompute reference"
    );
}

/// Quantized caches are lossy by construction, so they are not required to match
/// token-for-token — but the prefill logits must stay close, and int8 must be
/// closer than int4.
fn quantized_caches_stay_close(dir: &str) {
    let Some((model, tok)) = load(dir) else {
        return;
    };
    let model = model.as_ref();
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let logits_for = |quant| {
        let mut generator = Generator::with_quant(model, quant);
        generator
            .prefill(&ids)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .expect("logits row")
    };

    let full = logits_for(KvQuant::None);
    let int8 = logits_for(KvQuant::Int8);
    let int4 = logits_for(KvQuant::Int4);

    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // Absolute logit deltas are not comparable across models: Qwen3's logits
    // span a far wider range than GPT-2's, so the same *relative* quantization
    // error surfaces as a larger absolute number. Scale by the spread of the
    // reference logits so the threshold means the same thing on both.
    let mean = full.iter().sum::<f32>() / full.len() as f32;
    let std = (full.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / full.len() as f32).sqrt();

    let d8 = max_abs(&full, &int8) / std;
    let d4 = max_abs(&full, &int4) / std;
    eprintln!("{dir}: logit std {std:.3}, int8 drift {d8:.4} std, int4 drift {d4:.4} std");

    // Measured drift, prefill on the fixed prompt:
    //
    //   gpt2 124M       int8 0.038 std   int4 0.228 std
    //   qwen3 0.6B      int8 0.268 std   int4 3.525 std
    //
    // Qwen3 is ~7x more sensitive to int8 and ~15x more to int4 than GPT-2 —
    // int4 KV genuinely degrades it. The limits below are regression guards set
    // above the observed values, not a claim that either number is fine; the
    // architecture gap is a result in its own right, recorded in benchmarks/.
    assert!(
        d8 < 0.5,
        "{dir}: int8 prefill logits drifted {d8:.4} std (limit 0.5)"
    );
    assert!(
        d4 < 5.0,
        "{dir}: int4 prefill logits drifted {d4:.4} std (limit 5.0)"
    );
    assert!(
        d8 < d4,
        "{dir}: int8 ({d8:.4} std) should be closer than int4 ({d4:.4} std)"
    );

    // Whether the drift actually changes the emitted token is the thing that
    // matters, and it is architecture-dependent: int8 preserves GPT-2's argmax
    // on this prompt and flips Qwen3's. Reported, not asserted — pinning either
    // outcome would encode an accident of this prompt as a requirement.
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0usize, |b, (i, x)| if *x > v[b] { i } else { b })
    };
    eprintln!(
        "{dir}: argmax full={} int8={} int4={} (int8 {} top-1)",
        argmax(&full),
        argmax(&int8),
        argmax(&int4),
        if argmax(&full) == argmax(&int8) {
            "preserves"
        } else {
            "CHANGES"
        }
    );
}

/// Feeding the prompt one token at a time must land in the same place as one
/// bulk prefill — the chunked-prefill path exercises the offset causal mask and,
/// for a rotary model, the RoPE position offset.
fn incremental_prefill_matches_bulk(dir: &str) {
    let Some((model, tok)) = load(dir) else {
        return;
    };
    let model = model.as_ref();
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let row = |g: &Generator| {
        g.logits()
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };

    let mut bulk = Generator::new(model);
    bulk.prefill(&ids).unwrap();

    let mut stepwise = Generator::new(model);
    stepwise.prefill(&ids[..1]).unwrap();
    for &id in &ids[1..] {
        stepwise.feed(id).unwrap();
    }

    assert_eq!(bulk.len(), stepwise.len(), "{dir}: cache lengths differ");
    let (a, b) = (row(&bulk), row(&stepwise));
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 5e-2,
        "{dir}: token-by-token prefill diverged from bulk prefill: max|d| = {max_abs}"
    );
}

/// Under grouped-query attention the cache must hold `n_kv_head` heads, not
/// `n_head` — storing the expanded copy would silently multiply the footprint by
/// the group count while still producing correct logits.
fn cache_stores_kv_heads_not_query_heads(dir: &str) {
    let Some((model, tok)) = load(dir) else {
        return;
    };
    let model = model.as_ref();
    let meta = model.meta().clone();
    let ids = tok.encode(PROMPT, true).unwrap().get_ids().to_vec();

    let mut cache = model.new_cache();
    let input = candle_core::Tensor::from_vec(ids.clone(), (1, ids.len()), model.device()).unwrap();
    model.forward_with_cache(&input, &mut cache).unwrap();

    let keys = cache.layer(0).keys().unwrap().unwrap();
    assert_eq!(
        keys.dims(),
        &[1, meta.n_kv_head, ids.len(), meta.head_dim],
        "{dir}: cached key shape should be [batch, n_kv_head, seq, head_dim]"
    );
}

#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn gpt2_cache_matches_no_cache() {
    cache_matches_no_cache(GPT2_DIR);
}

#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn gpt2_quantized_caches_stay_close_to_fp32() {
    quantized_caches_stay_close(GPT2_DIR);
}

#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn gpt2_incremental_prefill_matches_bulk() {
    incremental_prefill_matches_bulk(GPT2_DIR);
}

#[test]
#[ignore = "requires benchmarks/gpt2 weights on disk"]
fn gpt2_cache_shape_matches_head_counts() {
    cache_stores_kv_heads_not_query_heads(GPT2_DIR);
}

#[test]
#[ignore = "requires benchmarks/qwen3-0.6b weights on disk"]
fn qwen3_cache_matches_no_cache() {
    cache_matches_no_cache(QWEN3_DIR);
}

#[test]
#[ignore = "requires benchmarks/qwen3-0.6b weights on disk"]
fn qwen3_quantized_caches_stay_close_to_bf16() {
    quantized_caches_stay_close(QWEN3_DIR);
}

#[test]
#[ignore = "requires benchmarks/qwen3-0.6b weights on disk"]
fn qwen3_incremental_prefill_matches_bulk() {
    incremental_prefill_matches_bulk(QWEN3_DIR);
}

#[test]
#[ignore = "requires benchmarks/qwen3-0.6b weights on disk"]
fn qwen3_cache_stores_kv_heads_not_query_heads() {
    cache_stores_kv_heads_not_query_heads(QWEN3_DIR);
}
