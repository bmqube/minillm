//! Minimal library-usage example.
//!
//! ```text
//! cargo run --release --example generate -- "Your prompt here"
//! ```

use minillm::generation::{Generator, SamplingConfig};
use minillm::{device, loader};

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

    let mut generator = Generator::new(&model, &dev);
    generator.prefill(&ids)?;

    print!("{prompt}");
    for _ in 0..40 {
        let next = generator.next_token(&cfg)?;
        if next == eot {
            break;
        }
        print!("{}", tokenizer.decode(&[next], false)?);
    }
    println!();
    Ok(())
}
