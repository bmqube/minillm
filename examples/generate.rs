//! Minimal library-usage example.
//!
//! ```text
//! cargo run --release --example generate -- "Your prompt here"
//! ```

use minillm::generation::{Generator, SamplingConfig};
use minillm::{device, loader};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The future of AI is".to_string());

    let dev = device::best();
    let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;

    let ids = tokenizer.encode(prompt.as_str(), true)?.get_ids().to_vec();
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX);
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
    };

    let mut generator = Generator::new(model.as_ref());
    generator.prefill(&ids)?;

    print!("{prompt}");
    // Each step feeds the token it returns, so stop while that still fits.
    let steps = 40.min(model.meta().n_ctx.saturating_sub(ids.len()));
    // Holds back a character split across tokens until it is whole.
    let mut text = tokenizer.decode_stream(false);
    for _ in 0..steps {
        let next = generator.next_token(&cfg)?;
        if next == eot {
            break;
        }
        if let Some(piece) = text.step(next)? {
            print!("{piece}");
            std::io::stdout().flush()?;
        }
    }
    println!();
    Ok(())
}
