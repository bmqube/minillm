use candle_core::Tensor;
use minillm::generation::{self, SamplingConfig};
use minillm::kv_cache::KvCache;
use minillm::{device, loader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let device = device::best();
    eprintln!("device: {device:?}");

    let (model, tokenizer) = loader::load("openai-community/gpt2", &device)?;

    let prompt = "The future of AI is";
    let mut input_ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    let max_new_tokens = 50;
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX);
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
    };

    println!("Input: {prompt}");
    print!("Generated: ");

    // Prefill: run the whole prompt through the model once, seeding the cache.
    let mut cache = KvCache::new(model.config().n_layer);
    let prompt_tensor = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?;
    let mut logits = model.forward_with_cache(&prompt_tensor, &mut cache)?;

    let start = Instant::now();
    let mut generated = 0usize;
    for _ in 0..max_new_tokens {
        let next = generation::sample(&logits, &cfg)?;
        if next == eot {
            break;
        }

        print!("{}", tokenizer.decode(&[next], false)?);
        input_ids.push(next);
        generated += 1;

        // Decode: feed just the new token; the cache holds the rest.
        let step = Tensor::from_vec(vec![next], (1, 1), &device)?;
        logits = model.forward_with_cache(&step, &mut cache)?;
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
