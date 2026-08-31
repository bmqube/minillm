use minillm::generation::{Generator, SamplingConfig};
use minillm::{device, loader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let device = device::best();
    eprintln!("device: {device:?}");

    let (model, tokenizer) = loader::load("openai-community/gpt2", &device)?;

    let prompt = "The future of AI is";
    let ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    let max_new_tokens = 50;
    let eot = tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX);
    let cfg = SamplingConfig {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
    };

    println!("Input: {prompt}");
    print!("Generated: ");

    // Prefill the prompt once, then decode one token per step against the cache.
    let mut generator = Generator::new(&model, &device);
    generator.prefill(&ids)?;

    let start = Instant::now();
    let mut generated = 0usize;
    for _ in 0..max_new_tokens {
        let next = generator.next_token(&cfg)?;
        if next == eot {
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
