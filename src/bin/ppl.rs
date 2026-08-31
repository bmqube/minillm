//! Sliding-window language-model perplexity for MiniLLM.
//!
//! ```text
//! cargo run --release --bin ppl -- [MODEL] [TEXT_FILE] [WINDOW] [STRIDE] [MAX_TOKENS]
//! ```
//!
//! Defaults: `benchmarks/gpt2 benchmarks/wikitext2.txt 512 256 8192`.
//!
//! `MODEL` is a local directory (with `config.json`, `tokenizer.json`,
//! `model.safetensors`) or a Hub id like `openai-community/gpt2`. `TEXT_FILE` is
//! raw UTF-8 text (e.g. the WikiText-2 raw test split concatenated into one
//! file). `WINDOW` is the context length per forward pass, `STRIDE` how far the
//! window advances between passes (`STRIDE < WINDOW` gives every scored token at
//! least `WINDOW - STRIDE` tokens of left context), and `MAX_TOKENS` caps the
//! number of tokens evaluated so a run stays short.
//!
//! Each target token is scored exactly once, by the window that has the most
//! left context for it. Negative log-likelihood is accumulated on the CPU from
//! the raw logits (`logsumexp(row) - row[target]`) — the same scalar,
//! obviously-correct style as `generation.rs`.
//!
//! There is no KV cache, so every window is a full `O(window^2)` attention
//! recompute. That is fine for a once-off evaluation; a `--cache` mode is
//! planned so the cache path can be perplexity-checked too (it must not change
//! the number).
//!
//! Auto-discovered by cargo as the `ppl` binary (like `bench` / `parity_dump`),
//! so no `[[bin]]` entry in `Cargo.toml` is needed.
//!
//! Sanity: fp32 `openai-community/gpt2` (124M) on the WikiText-2 raw test split
//! with `WINDOW = 512`, `STRIDE = 256` gives perplexity ~29-30. A wildly
//! different value means the harness is wrong (windowing, off-by-one on targets,
//! or stray special tokens).

use candle_core::{DType, IndexOp, Tensor};
use minillm::{device, loader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let model_id = args.next().unwrap_or_else(|| "benchmarks/gpt2".to_string());
    let text_path = args
        .next()
        .unwrap_or_else(|| "benchmarks/wikitext2.txt".to_string());
    let window: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let stride: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8192);

    assert!(window >= 2, "WINDOW must be >= 2");
    assert!(stride >= 1 && stride <= window, "need 1 <= STRIDE <= WINDOW");

    let dev = device::best();
    eprintln!("device : {dev:?}");
    eprintln!("model  : {model_id}");
    eprintln!("text   : {text_path}");
    eprintln!("window : {window}   stride : {stride}   max_tokens : {max_tokens}");

    let (model, tokenizer) = loader::load(&model_id, &dev)?;

    let text = std::fs::read_to_string(&text_path).map_err(|e| {
        format!("could not read {text_path}: {e}\n  see benchmarks/README.md for how to fetch WikiText-2")
    })?;

    // No special tokens: this is raw LM scoring over a continuous stream.
    let mut tokens: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    if tokens.len() > max_tokens {
        tokens.truncate(max_tokens);
    }
    let n = tokens.len();
    eprintln!("tokens : {n}");
    assert!(n >= 2, "need at least 2 tokens to score anything");

    let ctx = model.config().n_ctx;
    assert!(
        window <= ctx,
        "WINDOW {window} exceeds model context {ctx}"
    );

    let t0 = Instant::now();
    let mut nll = 0.0f64; // sum of -log p(target) over all scored tokens
    let mut scored = 0usize;
    let mut windows = 0usize;

    let mut start = 0usize;
    let mut prev_end = 0usize; // absolute index one past the last already-scored target
    loop {
        let end = (start + window).min(n);
        let len = end - start;
        if len < 2 {
            break;
        }

        let input = Tensor::from_vec(tokens[start..end].to_vec(), (1, len), &dev)?;
        let logits = model.forward(&input)?.i(0)?.to_dtype(DType::F32)?; // [len, vocab]

        // Score every target token whose absolute index is in `prev_end..end`
        // (each target is handled by exactly one window). Target `abs_t` is
        // predicted from row `abs_t - start - 1` of this window's logits.
        let first_target = prev_end.max(start + 1);
        for (offset, &target_id) in tokens[first_target..end].iter().enumerate() {
            let abs_t = first_target + offset;
            let row: Vec<f32> = logits.i(abs_t - start - 1)?.to_vec1::<f32>()?;
            nll += (logsumexp(&row) - row[target_id as usize]) as f64;
            scored += 1;
        }

        prev_end = end;
        windows += 1;
        if end == n {
            break;
        }
        start += stride;
    }

    let secs = t0.elapsed().as_secs_f64();
    let mean_nll = nll / scored as f64;
    let ppl = mean_nll.exp();
    let bits_per_token = mean_nll / std::f64::consts::LN_2;

    println!();
    println!("windows scored      : {windows}");
    println!("tokens scored       : {scored}");
    println!("mean NLL (nats/tok) : {mean_nll:.5}");
    println!("bits / token        : {bits_per_token:.5}");
    println!("perplexity          : {ppl:.4}");
    println!("eval time           : {secs:.1} s");

    Ok(())
}

/// `log(sum(exp(v)))`, shifted by the max for numerical stability.
fn logsumexp(v: &[f32]) -> f32 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|&x| (x - m).exp()).sum();
    m + sum.ln()
}
