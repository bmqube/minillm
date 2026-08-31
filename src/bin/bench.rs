//! Throughput + size benchmark for MiniLLM.
//!
//! ```text
//! cargo run --release --bin bench -- [MODEL] [PREFILL_TOKENS] [DECODE_TOKENS]
//! ```
//!
//! Defaults: `benchmarks/gpt2 64 128`. `MODEL` is a local directory (with
//! `config.json`, `tokenizer.json`, `model.safetensors`) or a Hub id like
//! `openai-community/gpt2`.
//!
//! Reports model load time, analytic parameter count and fp32 weight memory,
//! prefill latency, and greedy decode throughput **both without and with the KV
//! cache**, plus the speedup between them. The no-cache loop recomputes the whole
//! sequence every step (O(n^2)); the cached loop feeds one token per step.

use candle_core::{IndexOp, Tensor};
use minillm::kv_cache::KvCache;
use minillm::{device, loader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let model_id = args.next().unwrap_or_else(|| "benchmarks/gpt2".to_string());
    let prefill: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let decode: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);

    let dev = device::best();
    println!("device         : {dev:?}");
    println!("model          : {model_id}");
    println!("prefill tokens : {prefill}");
    println!("decode steps   : {decode}");

    let t0 = Instant::now();
    let (model, _tok) = loader::load(&model_id, &dev)?;
    println!("load time      : {:.2} s", t0.elapsed().as_secs_f64());

    let params = model.config().num_parameters();
    println!("parameters     : {params} (~{:.1} M)", params as f64 / 1e6);
    println!(
        "fp32 weights   : ~{:.2} GiB",
        (params as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
    );

    // Deterministic pseudo-prompt: ids 0..prefill.
    let ids: Vec<u32> = (0..prefill as u32).collect();

    // Warm-up: triggers lazy allocation / kernel selection.
    let warm = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let _ = model.forward(&warm)?.i((0, ids.len() - 1))?.to_vec1::<f32>()?;

    // Prefill (full forward pass over the prompt).
    let t1 = Instant::now();
    let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let logits = model.forward(&input)?;
    let _ = logits.i((0, ids.len() - 1))?.to_vec1::<f32>()?; // force realisation
    let prefill_s = t1.elapsed().as_secs_f64();
    println!(
        "prefill        : {:.1} ms ({:.1} tok/s)",
        prefill_s * 1e3,
        prefill as f64 / prefill_s
    );

    // --- Decode without a KV cache: recompute the whole sequence every step. ---
    let mut seq = ids.clone();
    let t2 = Instant::now();
    for _ in 0..decode {
        let input = Tensor::from_vec(seq.clone(), (1, seq.len()), &dev)?;
        let row = model.forward(&input)?.i((0, seq.len() - 1))?.to_vec1::<f32>()?;
        seq.push(argmax(&row) as u32);
    }
    let nocache_s = t2.elapsed().as_secs_f64();
    let nocache_tps = decode as f64 / nocache_s;
    println!(
        "decode(no cache): {nocache_tps:.2} tok/s  ({decode} tokens in {nocache_s:.2} s, seq {prefill} -> {})",
        prefill + decode
    );

    // --- Decode with a KV cache: prefill once, then one token per step. ---
    let mut cache = KvCache::new(model.config().n_layer);
    let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let prime = model.forward_with_cache(&input, &mut cache)?;
    let mut next = argmax(&prime.i((0, ids.len() - 1))?.to_vec1::<f32>()?) as u32;

    let t3 = Instant::now();
    for _ in 0..decode {
        let step = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        let row = model
            .forward_with_cache(&step, &mut cache)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
        next = argmax(&row) as u32;
    }
    let cache_s = t3.elapsed().as_secs_f64();
    let cache_tps = decode as f64 / cache_s;
    println!(
        "decode(KV cache): {cache_tps:.2} tok/s  ({decode} tokens in {cache_s:.2} s, seq {prefill} -> {})",
        prefill + decode
    );
    println!("speedup        : {:.1}x", cache_tps / nocache_tps);

    println!();
    println!("peak RAM: measure externally, e.g.");
    println!("  Linux  : /usr/bin/time -v <cmd>   -> 'Maximum resident set size'");
    println!("  macOS  : /usr/bin/time -l <cmd>   -> 'maximum resident set size'");
    println!("  Windows: Get-Process bench | Select-Object PeakWorkingSet64");

    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
