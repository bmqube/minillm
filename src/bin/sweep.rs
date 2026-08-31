//! Sequence-length x model-size throughput sweep for MiniLLM.
//!
//! ```text
//! cargo run --release --bin sweep -- [MODELS] [PREFILLS] [DECODE] [CACHE] [KVQUANT] [--repeats N]
//!     > benchmarks/sweep_cpu.csv
//! ```
//!
//! Defaults: `benchmarks/gpt2  32,64,128,256,512  64  both  none  --repeats 1`.
//!
//! - `MODELS`    comma-separated local dirs (with `config.json`, `tokenizer.json`,
//!   `model.safetensors`) or Hub ids like `openai-community/gpt2`.
//! - `PREFILLS`  comma-separated prompt lengths to test.
//! - `DECODE`    greedy decode steps run after each prefill.
//! - `CACHE`     `off`, `on`, or `both` — which decode paths to measure.
//! - `KVQUANT`   `none` (fp32), `int8`, `int4` — how the `on` path stores K/V.
//! - `--repeats` how many timed trials per configuration (default 1).
//!
//! Emits **one CSV row per trial** — `(model, prefill, kv_cache, repeat)` — to
//! stdout, so downstream analysis can compute its own error bars; a mean ± std
//! summary is printed to stderr. Decode throughput on a loaded CPU varies
//!15-20% run to run, so any headline number should come from `--repeats 5` or
//! more, not a single trial.

use candle_core::{IndexOp, Tensor};
use minillm::generation::{greedy_no_cache, Generator, GREEDY};
use minillm::kv_cache::KvQuant;
use minillm::{device, loader, model::GPT2Model};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    // Pull `--repeats N` out, then read the positionals.
    let mut positional: Vec<String> = Vec::new();
    let mut repeats: usize = 1;
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
    eprintln!("kv_quant     : {kv_quant:?}");
    eprintln!("repeats      : {repeats}");

    println!(
        "model,params,kv_bytes_per_token,device,dtype,kv_cache,kv_quant,repeat,\
         prefill_tokens,decode_steps,seq_start,seq_end,load_s,prefill_ms,\
         prefill_tok_s,decode_tok_s,decode_s"
    );

    // config label -> decode tok/s across trials, for the stderr summary.
    let mut trials: BTreeMap<String, Vec<f64>> = BTreeMap::new();

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
        let cfg = model.config();
        let params = cfg.num_parameters();
        let head_dim = cfg.n_embd / cfg.n_head;
        let kv_bpt_fp32 = cfg.kv_cache_bytes_per_token(4);
        let kv_bpt_on = kv_quant.bytes_per_token(cfg.n_layer, cfg.n_head, head_dim);
        eprintln!(
            "  loaded in {load_s:.2} s, {params} params, KV {kv_bpt_fp32} B/token fp32, \
             {kv_bpt_on} B/token {kv_quant:?}"
        );

        for &prefill in &prefills {
            if prefill < 1 {
                continue;
            }
            let ids: Vec<u32> = (0..prefill as u32).collect();

            // Warm-up (lazy alloc / kernel selection) — not timed, once per config.
            let warm = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            let _ = model
                .forward(&warm)?
                .i((0, ids.len() - 1))?
                .to_vec1::<f32>()?;

            for repeat in 1..=repeats {
                // Prefill (full forward over the prompt), timed per trial.
                let t1 = Instant::now();
                let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
                let _ = model
                    .forward(&input)?
                    .i((0, ids.len() - 1))?
                    .to_vec1::<f32>()?;
                let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
                let prefill_tok_s = prefill as f64 / (prefill_ms / 1e3);

                let emit =
                    |kv: &str, quant: &str, kv_bpt: usize, decode_tok_s: f64, decode_s: f64| {
                        println!(
                            "{model_id},{params},{kv_bpt},{dev_str},f32,{kv},{quant},{repeat},\
                             {prefill},{decode},{prefill},{},{load_s:.3},{prefill_ms:.1},\
                             {prefill_tok_s:.2},{decode_tok_s:.3},{decode_s:.3}",
                            prefill + decode
                        );
                    };

                if do_off {
                    let (tps, s) = time_decode_no_cache(&model, &ids, decode, &dev)?;
                    emit("off", "none", kv_bpt_fp32, tps, s);
                    trials
                        .entry(format!("{model_id} prefill={prefill} off"))
                        .or_default()
                        .push(tps);
                }
                if do_on {
                    let (tps, s) = time_decode_cached(&model, &ids, decode, kv_quant, &dev)?;
                    emit(
                        "on",
                        &format!("{kv_quant:?}").to_lowercase(),
                        kv_bpt_on,
                        tps,
                        s,
                    );
                    trials
                        .entry(format!("{model_id} prefill={prefill} on/{kv_quant:?}"))
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
    model: &GPT2Model,
    prompt: &[u32],
    steps: usize,
    dev: &candle_core::Device,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let t = Instant::now();
    greedy_no_cache(model, dev, prompt, steps)?;
    let s = t.elapsed().as_secs_f64();
    Ok((steps as f64 / s, s))
}

/// Time `steps` greedy tokens against a KV cache. The prefill seeds the cache
/// and is deliberately excluded from the timing.
fn time_decode_cached(
    model: &GPT2Model,
    prompt: &[u32],
    steps: usize,
    quant: KvQuant,
    dev: &candle_core::Device,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let mut generator = Generator::with_quant(model, dev, quant);
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
