//! CLI demo: load a checkpoint and stream a completion.
//!
//! ```text
//! cargo run --release                                    # gpt2, fp32
//! cargo run --release -- Qwen/Qwen3-0.6B "Once upon a"    # qwen3, bf16
//! ```
//!
//! Arguments: `[MODEL] [PROMPT]`. `MODEL` is a Hub id or a local directory;
//! precision follows the checkpoint's own `torch_dtype`.

use minillm::generation::{Generator, SamplingConfig};
use minillm::{loader, ExecutionPlan};
use std::io::Write;
use std::time::Instant;

const DEFAULT_MODEL: &str = "openai-community/gpt2";
const DEFAULT_PROMPT: &str = "The future of AI is";

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let model_id = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let plan = ExecutionPlan::detect();
    plan.log();
    let device = plan.device().clone();

    let (model, tokenizer) = loader::load(&model_id, &device)?;
    let meta = model.meta();
    eprintln!(
        "model : {model_id} ({}, {} params, {}, {}/{} q/kv heads)",
        meta.architecture,
        meta.n_params,
        model.precision(),
        meta.n_head,
        meta.n_kv_head,
    );

    let ids = tokenizer.encode(prompt.as_str(), true)?.get_ids().to_vec();
    // Each step feeds the token it returns, so stop while that still fits.
    let max_new_tokens = 50.min(meta.n_ctx.saturating_sub(ids.len()));
    // Stop on whichever end markers this tokenizer has: GPT-2 ends on
    // <|endoftext|>, Qwen chat models on <|im_end|>.
    let stops: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
    };

    println!("Input: {prompt}");
    print!("Generated: ");

    // Prefill the prompt once, then decode one token per step against the cache.
    let mut generator = Generator::new(model.as_ref());
    generator.prefill(&ids)?;

    let start = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    for _ in 0..max_new_tokens {
        let next = generator.next_token(&cfg)?;
        if stops.contains(&next) {
            break;
        }
        out.push(next);
        // Byte-level BPE can split one character across tokens, so decode the
        // whole completion and print the new tail once it is whole.
        let text = tokenizer.decode(&out, false)?;
        if !text.ends_with('\u{FFFD}') {
            print!("{}", text.get(printed..).unwrap_or_default());
            std::io::stdout().flush()?;
            printed = text.len();
        }
    }
    println!();
    let generated = out.len();

    let secs = start.elapsed().as_secs_f64();
    if generated > 0 {
        eprintln!(
            "{generated} tokens in {secs:.2}s = {:.1} tok/s (KV cache)",
            generated as f64 / secs
        );
    }

    Ok(())
}
