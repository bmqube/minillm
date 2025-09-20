use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)?;

    // Encode some text
    let text = "Hey there! This is a mini inference engine.";
    let encoding = tokenizer.encode(text, false)?;

    println!("Original text: {}", text);
    println!("Tokens: {:?}", encoding.get_tokens());
    println!("Token IDs: {:?}", encoding.get_ids());

    // TODO: Implement model loading and inference
    // TODO: Add model execution pipeline

    Ok(())
}
