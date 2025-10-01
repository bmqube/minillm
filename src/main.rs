use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::new_cuda(0)?;
    let (model, tokenizer) = minillm::loader::load("openai-community/gpt2", &device)?;

    let prompt = "The future of AI is";
    let mut input_ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    let max_new_tokens = 50;

    println!("Input: {}", prompt);
    print!("Generated: ");

    for _ in 0..max_new_tokens {
        // Create tensor from current input_ids
        let batch_size = 1;
        let seq_len = input_ids.len();
        let input_tensor = Tensor::from_vec(input_ids.clone(), (batch_size, seq_len), &device)?;

        // Forward pass to get logits
        let logits = model.forward(&input_tensor)?;

        // Generate next token with some randomness (using only temperature for now)
        let next_token = minillm::generation::generate_token(&logits, 0.8, None, None)?;

        if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(u32::MAX) {
            break;
        }

        // Decode and print the new token
        let token_text = tokenizer.decode(&[next_token], false)?;
        print!("{}", token_text);

        input_ids.push(next_token);
    }

    println!(); // New line at the end

    Ok(())
}
