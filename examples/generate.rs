//! Minimal library-usage example.
//!
//! ```text
//! cargo run --release --example generate -- "Your prompt here"
//! ```

use candle_core::Tensor;
use minillm::generation::{sample, SamplingConfig};
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The future of AI is".to_string());

    let dev = device::best();
    let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;

    let mut ids = tokenizer.encode(prompt.as_str(), true)?.get_ids().to_vec();
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX);
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
    };

    print!("{prompt}");
    for _ in 0..40 {
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        let next = sample(&model.forward(&input)?, &cfg)?;
        if next == eot {
            break;
        }
        print!("{}", tokenizer.decode(&[next], false)?);
        ids.push(next);
    }
    println!();
    Ok(())
}
