//! Sequence-length x model-size throughput sweep for MiniLLM.
//!
//! ```text
//! cargo run --release --bin sweep -- [MODELS] [PREFILLS] [DECODE] [CACHE] > benchmarks/sweep_cpu.csv
//! ```
//!
//! Defaults: `benchmarks/gpt2  32,64,128,256,512  64  both`.
//!
//! - `MODELS`   comma-separated local dirs (with `config.json`, `tokenizer.json`,
//!   `model.safetensors`) or Hub ids like `openai-community/gpt2`.
//! - `PREFILLS` comma-separated prompt lengths to test.
//! - `DECODE`   greedy decode steps run after each prefill.
//! - `CACHE`    `off`, `on`, or `both` — which decode paths to measure.
//!
//! Emits one CSV row per `(model, prefill, kv_cache)` to stdout; progress goes to
//! stderr. `kv_cache=off` recomputes the whole sequence each step (O(n^2));
//! `kv_cache=on` prefills once then feeds one token per step. This is the data
//! generator for the paper's speedup-vs-sequence-length curves.

use candle_core::{IndexOp, Tensor};
use minillm::kv_cache::KvCache;
use minillm::{device, loader, model::GPT2Model};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let models: Vec<String> = split_csv(&args.next().unwrap_or_else(|| "benchmarks/gpt2".into()));
    let prefills: Vec<usize> = split_csv(&args.next().unwrap_or_else(|| "32,64,128,256,512".into()))
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let decode: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let cache_arg = args.next().unwrap_or_else(|| "both".into());
    let (do_off, do_on) = match cache_arg.as_str() {
        "off" => (true, false),
        "on" => (false, true),
        _ => (true, true),
    };

    let dev = device::best();
    let dev_str = format!("{dev:?}");
    eprintln!("device       : {dev_str}");
    eprintln!("models       : {models:?}");
    eprintln!("prefills     : {prefills:?}");
    eprintln!("decode steps : {decode}");
    eprintln!("cache        : {cache_arg}");

    println!(
        "model,params,device,dtype,kv_cache,prefill_tokens,decode_steps,\
         seq_start,seq_end,load_s,prefill_ms,prefill_tok_s,decode_tok_s,decode_s"
    );

    for model_id in &models {
        eprintln!("\n=== {model_id} ===");
        let t0 = Instant::now();
        let (model, _tok) = match loader::load(model_id, &dev) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  load failed: {e}  (skipping)");
                continue;
            }
        };
        let load_s = t0.elapsed().as_secs_f64();
        let params = model.config().num_parameters();
        eprintln!("  loaded in {load_s:.2} s, {params} params");

        for &prefill in &prefills {
            if prefill < 1 {
                continue;
            }
            let ids: Vec<u32> = (0..prefill as u32).collect();

            // Warm-up (lazy alloc / kernel selection) — not timed.
            let warm = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            let _ = model.forward(&warm)?.i((0, ids.len() - 1))?.to_vec1::<f32>()?;

            // Prefill (full forward over the prompt).
            let t1 = Instant::now();
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            let _ = model
                .forward(&input)?
                .i((0, ids.len() - 1))?
                .to_vec1::<f32>()?;
            let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
            let prefill_tok_s = prefill as f64 / (prefill_ms / 1e3);

            let emit = |kv: &str, decode_tok_s: f64, decode_s: f64| {
                println!(
                    "{model_id},{params},{dev_str},f32,{kv},{prefill},{decode},\
                     {prefill},{},{load_s:.3},{prefill_ms:.1},\
                     {prefill_tok_s:.2},{decode_tok_s:.3},{decode_s:.3}",
                    prefill + decode
                );
            };

            if do_off {
                let (tps, s) = decode_no_cache(&model, &ids, decode, &dev)?;
                eprintln!("  prefill {prefill:>4}  off  {tps:.2} tok/s");
                emit("off", tps, s);
            }
            if do_on {
                let (tps, s) = decode_with_cache(&model, &ids, decode, &dev)?;
                eprintln!("  prefill {prefill:>4}  on   {tps:.2} tok/s");
                emit("on", tps, s);
            }
        }
    }

    Ok(())
}

/// Greedy decode `steps` tokens, recomputing the full sequence each step.
fn decode_no_cache(
    model: &GPT2Model,
    prompt: &[u32],
    steps: usize,
    dev: &candle_core::Device,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let mut seq = prompt.to_vec();
    let t = Instant::now();
    for _ in 0..steps {
        let input = Tensor::from_vec(seq.clone(), (1, seq.len()), dev)?;
        let row = model.forward(&input)?.i((0, seq.len() - 1))?.to_vec1::<f32>()?;
        seq.push(argmax(&row) as u32);
    }
    let s = t.elapsed().as_secs_f64();
    Ok((steps as f64 / s, s))
}

/// Greedy decode `steps` tokens: prefill once, then one token per step.
fn decode_with_cache(
    model: &GPT2Model,
    prompt: &[u32],
    steps: usize,
    dev: &candle_core::Device,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let mut cache = KvCache::new(model.config().n_layer);
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let prime = model.forward_with_cache(&input, &mut cache)?;
    let mut next = argmax(&prime.i((0, prompt.len() - 1))?.to_vec1::<f32>()?) as u32;

    let t = Instant::now();
    for _ in 0..steps {
        let step = Tensor::from_vec(vec![next], (1, 1), dev)?;
        let row = model
            .forward_with_cache(&step, &mut cache)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
        next = argmax(&row) as u32;
    }
    let s = t.elapsed().as_secs_f64();
    Ok((steps as f64 / s, s))
}

fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
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
