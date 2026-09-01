//! Sequence-length x model-size throughput sweep for MiniLLM.
//!
//! ```text
//! cargo run --release --bin sweep -- [MODELS] [PREFILLS] [DECODE] [CACHE] [KVQUANT]
//!     [--repeats N] [--precision f32|f16|bf16] > benchmarks/sweep_cpu.csv
//! ```
//!
//! Defaults: `benchmarks/gpt2  32,64,128,256,512  64  both  none  --repeats 1`.
//!
//! - `MODELS`    comma-separated local dirs (with `config.json`, `tokenizer.json`,
//!   and either `model.safetensors` or a shard index) or Hub ids like
//!   `openai-community/gpt2` / `Qwen/Qwen3-0.6B`.
//! - `PREFILLS`  comma-separated prompt lengths to test.
//! - `DECODE`    greedy decode steps run after each prefill.
//! - `CACHE`     `off`, `on`, or `both` — which decode paths to measure.
//! - `KVQUANT`   `none`, `int8`, `int4` — how the `on` path stores K/V.
//! - `--repeats` how many timed trials per configuration (default 1).
//! - `--precision` pin the weight dtype; default is the checkpoint's own
//!   `torch_dtype`, so GPT-2 stays fp32 and Qwen3 loads bf16.
//!
//! Emits **one CSV row per trial** — `(model, prefill, kv_cache, repeat)` — to
//! stdout, so downstream analysis can compute its own error bars; a mean ± std
//! summary is printed to stderr. Decode throughput on a loaded CPU varies
//!15-20% run to run, so any headline number should come from `--repeats 5` or
//! more, not a single trial.

use candle_core::{IndexOp, Tensor};
use minillm::dtype::Precision;
use minillm::generation::{greedy_no_cache, Generator, GREEDY};
use minillm::kv_cache::KvQuant;
use minillm::models::CausalLM;
use minillm::{device, loader};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    // Pull the flags out, then read the positionals.
    let mut positional: Vec<String> = Vec::new();
    let mut repeats: usize = 1;
    let mut precision: Option<Precision> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--repeats" => {
                let v = it.next().unwrap_or_default();
                repeats = v
                    .parse()
                    .unwrap_or_else(|_| panic!("--repeats expects a number, got {v:?}"));
                assert!(repeats >= 1, "--repeats must be >= 1");
            }
            "--precision" => {
                let v = it.next().unwrap_or_default();
                precision = Some(
                    Precision::parse(&v)
                        .unwrap_or_else(|| panic!("--precision expects f32|f16|bf16, got {v:?}")),
                );
            }
            _ => positional.push(a),
        }
    }
    let mut args = positional.into_iter();

    let models: Vec<String> = split_csv(&args.next().unwrap_or_else(|| "benchmarks/gpt2".into()));
    let prefills: Vec<usize> =
        split_csv(&args.next().unwrap_or_else(|| "32,64,128,256,512".into()))
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
    let kvq_arg = args.next().unwrap_or_else(|| "none".into());
    let kv_quant = KvQuant::parse(&kvq_arg)
        .unwrap_or_else(|| panic!("KVQUANT expects none|int8|int4, got {kvq_arg:?}"));

    let dev = device::best();
    let dev_str = format!("{dev:?}");
    eprintln!("device       : {dev_str}");
    eprintln!("models       : {models:?}");
    eprintln!("prefills     : {prefills:?}");
    eprintln!("decode steps : {decode}");
    eprintln!("cache        : {cache_arg}");
    eprintln!("kv_quant     : {kv_quant}");
    eprintln!("repeats      : {repeats}");
    eprintln!(
        "precision    : {}",
        precision.map_or("checkpoint default".to_string(), |p| p.to_string())
    );

    println!(
        "model,arch,params,n_kv_head,kv_bytes_per_token,device,dtype,kv_cache,kv_quant,repeat,\
         prefill_tokens,decode_steps,seq_start,seq_end,load_s,prefill_ms,\
         prefill_tok_s,decode_tok_s,decode_s"
    );

    // config label -> decode tok/s across trials, for the stderr summary.
    let mut trials: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for model_id in &models {
        eprintln!("\n=== {model_id} ===");
        let t0 = Instant::now();
        let (model, _tok) = match loader::load_with(model_id, &dev, precision) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  load failed: {e}  (skipping)");
                continue;
            }
        };
        let load_s = t0.elapsed().as_secs_f64();
        let model = model.as_ref();
        let meta = model.meta().clone();
        let dtype = model.precision();
        let params = meta.n_params;
        let arch = meta.architecture;
        let n_kv_head = meta.n_kv_head;
        // The `off` rows carry the unquantized cost so both rows of a pair stay
        // comparable against the same baseline.
        let kv_bpt_full = meta.kv_cache_bytes_per_token_at(dtype);
        let kv_bpt_on = meta.kv_cache_bytes_per_token_with(kv_quant, dtype);
        eprintln!(
            "  loaded in {load_s:.2} s, {arch}, {params} params, {dtype}, \
             {}/{} q/kv heads, KV {kv_bpt_full} B/token {dtype}, \
             {kv_bpt_on} B/token {kv_quant}",
            meta.n_head, meta.n_kv_head
        );

        for &prefill in &prefills {
            if prefill < 1 {
                continue;
            }
            let ids: Vec<u32> = (0..prefill as u32).collect();

            // Warm-up (lazy alloc / kernel selection) — not timed, once per config.
            let warm = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            let _ = model.forward_last(&warm)?.i(0)?.to_vec1::<f32>()?;

            for repeat in 1..=repeats {
                // Prefill (full forward over the prompt, only the final position's
                // logits kept via `forward_last` — what a real decode session
                // actually needs), timed per trial.
                let t1 = Instant::now();
                let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
                let _ = model.forward_last(&input)?.i(0)?.to_vec1::<f32>()?;
                let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
                let prefill_tok_s = prefill as f64 / (prefill_ms / 1e3);

                let emit =
                    |kv: &str, quant: &str, kv_bpt: usize, decode_tok_s: f64, decode_s: f64| {
                        println!(
                            "{model_id},{arch},{params},{n_kv_head},{kv_bpt},{dev_str},{dtype},\
                             {kv},{quant},{repeat},{prefill},{decode},{prefill},{},\
                             {load_s:.3},{prefill_ms:.1},{prefill_tok_s:.2},\
                             {decode_tok_s:.3},{decode_s:.3}",
                            prefill + decode
                        );
                    };

                if do_off {
                    let (tps, s) = time_decode_no_cache(model, &ids, decode)?;
                    emit("off", "none", kv_bpt_full, tps, s);
                    trials
                        .entry(format!("{model_id} prefill={prefill} off"))
                        .or_default()
                        .push(tps);
                }
                if do_on {
                    let (tps, s) = time_decode_cached(model, &ids, decode, kv_quant)?;
                    emit("on", kv_quant.as_str(), kv_bpt_on, tps, s);
                    trials
                        .entry(format!("{model_id} prefill={prefill} on/{kv_quant}"))
                        .or_default()
                        .push(tps);
                }
            }
            eprintln!("  prefill {prefill:>4}  done ({repeats} trial(s))");
        }
    }

    eprintln!("\n=== decode tok/s, mean +/- std over {repeats} trial(s) ===");
    for (label, xs) in &trials {
        let (mean, sd) = mean_std(xs);
        let spread = if xs.len() > 1 {
            format!(" +/- {sd:.2} ({:.1}%)", 100.0 * sd / mean)
        } else {
            String::new()
        };
        eprintln!("  {label:<48} {mean:.2}{spread}");
    }

    Ok(())
}

/// Time `steps` greedy tokens with no cache (full recompute each step).
fn time_decode_no_cache(
    model: &dyn CausalLM,
    prompt: &[u32],
    steps: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let t = Instant::now();
    greedy_no_cache(model, prompt, steps)?;
    let s = t.elapsed().as_secs_f64();
    Ok((steps as f64 / s, s))
}

/// Time `steps` greedy tokens against a KV cache. The prefill seeds the cache
/// and is deliberately excluded from the timing.
fn time_decode_cached(
    model: &dyn CausalLM,
    prompt: &[u32],
    steps: usize,
    quant: KvQuant,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let mut generator = Generator::with_quant(model, quant);
    generator.prefill(prompt)?;
    let t = Instant::now();
    for _ in 0..steps {
        generator.next_token(&GREEDY)?;
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

/// Sample mean and (n-1 denominator) standard deviation.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}
