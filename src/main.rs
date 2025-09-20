use hf_hub::api::sync::Api;
use serde_json::Value;
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());

    // Download model files
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_path = repo.get("model.safetensors")?;
    let config_path = repo.get("config.json")?;

    println!("Files downloaded:");
    println!("📁 Tokenizer: {:?}", tokenizer_path);
    println!("📁 Config: {:?}", config_path);
    println!("📁 Model: {:?}", model_path);

    //Load model config
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Value = serde_json::from_str(&config_str)?;

    println!("\nModel Config");
    let vocab_size = config.get("vocab_size").unwrap();
    let hidden_size = config.get("hidden_size").unwrap();
    let num_key_value_heads = config.get("num_key_value_heads").unwrap();
    let num_layers = config
        .get("num_hidden_layers")
        .or_else(|| config.get("num_layers"))
        .or_else(|| config.get("n_layer"))
        .or_else(|| config.get("n_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("Vocab Size: {}", vocab_size);
    println!("Hidden Size: {}", hidden_size);
    println!("Number of Layers: {}", num_layers);
    println!("Number of Key-Value Heads: {}", num_key_value_heads);

    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    // Encode some text
    let text = "Hey there! This is a mini inference engine.";
    let encoded_prompt = tokenizer.encode(text, false)?;
    let token_ids = encoded_prompt.get_ids();

    println!("Original text: {}", text);
    println!("Tokens: {:?}", encoded_prompt.get_tokens());
    println!("Token IDs: {:?}", token_ids);

    Ok(())
}
