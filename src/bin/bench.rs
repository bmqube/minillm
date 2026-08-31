//! Throughput + size benchmark for MiniLLM.
//!
//! ```text
//! cargo run --release --bin bench -- [MODEL_ID] [PREFILL_TOKENS] [DECODE_TOKENS]
//! ```
//!
//! Defaults: `openai-community/gpt2 64 128`.
//!
//! Reports model load time, analytic parameter count and fp32 weight memory,
//! prefill latency, and greedy decode throughput. There is no KV cache yet, so
//! every decode step recomputes the whole sequence: these numbers are the
//! baseline that the planned KV-cache work is measured against.

use candle_core::{IndexOp, Tensor};
use minillm::{device, loader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let model_id = args
        .next()
        .unwrap_or_else(|| "openai-community/gpt2".to_string());
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
    println!(
        "parameters     : {params} (~{:.1} M)",
        params as f64 / 1e6
    );
    println!(
        "fp32 weights   : ~{:.2} GiB",
        (params as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0)
    );

    // Deterministic pseudo-prompt: ids 0..prefill.
    let mut ids: Vec<u32> = (0..prefill as u32).collect();

    // Warm-up: triggers lazy allocation / kernel selection.
    let warm = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let _ = model.forward(&warm)?.i((0, ids.len() - 1))?.to_vec1::<f32>()?;

    // Prefill.
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

    // Greedy decode loop (no sampling cost, no KV cache).
    let t2 = Instant::now();
    for _ in 0..decode {
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        let row = model
            .forward(&input)?
            .i((0, ids.len() - 1))?
            .to_vec1::<f32>()?;
        let mut best = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
        }
        ids.push(best as u32);
    }
    let decode_s = t2.elapsed().as_secs_f64();
    println!(
        "decode         : {:.2} tok/s  ({decode} tokens in {decode_s:.2} s, seq {prefill} -> {})",
        decode as f64 / decode_s,
        prefill + decode
    );

    println!();
    println!("peak RAM: measure externally, e.g.");
    println!("  Linux  : /usr/bin/time -v <cmd>   -> 'Maximum resident set size'");
    println!("  macOS  : /usr/bin/time -l <cmd>   -> 'maximum resident set size'");
    println!("  Windows: Get-Process bench | Select-Object PeakWorkingSet64");

    Ok(())
}
