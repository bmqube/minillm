use candle_core::Tensor;
use minillm::generation::{self, SamplingConfig};
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

    let start = Instant::now();
    let mut generated = 0usize;
    for _ in 0..max_new_tokens {
        let seq_len = input_ids.len();
        let input = Tensor::from_vec(input_ids.clone(), (1, seq_len), &device)?;
        let logits = model.forward(&input)?;

        let next = generation::sample(&logits, &cfg)?;
        if next == eot {
            break;
        }

        print!("{}", tokenizer.decode(&[next], false)?);
        input_ids.push(next);
        generated += 1;
    }
    println!();

    let secs = start.elapsed().as_secs_f64();
    if generated > 0 {
        eprintln!(
            "{generated} tokens in {secs:.2}s = {:.1} tok/s (no KV cache)",
            generated as f64 / secs
        );
    }

    Ok(())
}
