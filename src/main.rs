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
use minillm::{device, loader};
use std::time::Instant;

const DEFAULT_MODEL: &str = "openai-community/gpt2";
const DEFAULT_PROMPT: &str = "The future of AI is";

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let model_id = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let device = device::best();
    eprintln!("device: {device:?}");

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
    let max_new_tokens = 50;
    // GPT-2 ends on <|endoftext|>; Qwen3 on <|im_end|>. Neither is present in
    // the other's vocabulary, so looking up both is safe.
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
    let mut generated = 0usize;
    for _ in 0..max_new_tokens {
        let next = generator.next_token(&cfg)?;
        if stops.contains(&next) {
            break;
        }
        print!("{}", tokenizer.decode(&[next], false)?);
        generated += 1;
    }
    println!();

    let secs = start.elapsed().as_secs_f64();
    if generated > 0 {
        eprintln!(
            "{generated} tokens in {secs:.2}s = {:.1} tok/s (KV cache)",
            generated as f64 / secs
        );
    }

    Ok(())
}
