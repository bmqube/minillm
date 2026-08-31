//! Minimal library-usage example.
//!
//! ```text
//! cargo run --release --example generate -- "Your prompt here"
//! ```

use candle_core::Tensor;
use minillm::generation::{sample, SamplingConfig};
use minillm::kv_cache::KvCache;
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

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

    // Prefill the prompt, then decode one token at a time against the KV cache.
    let mut cache = KvCache::new(model.config().n_layer);
    let prompt_tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let mut logits = model.forward_with_cache(&prompt_tensor, &mut cache)?;

    print!("{prompt}");
    for _ in 0..40 {
        let next = sample(&logits, &cfg)?;
        if next == eot {
            break;
        }
        print!("{}", tokenizer.decode(&[next], false)?);
        ids.push(next);

        let step = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        logits = model.forward_with_cache(&step, &mut cache)?;
    }
    println!();
    Ok(())
}
