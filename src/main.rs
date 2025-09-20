use hf_hub::api::sync::Api;
use serde_json::Value;
use tokenizers::Tokenizer;

fn main() {
    let api = Api::new().unwrap();
    let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());

    // Download model files
    let tokenizer_path = repo.get("tokenizer.json").unwrap();
    let model_path = repo.get("model.safetensors").unwrap();
    let config_path = repo.get("config.json").unwrap();

    println!("Files downloaded:");
    println!("📁 Tokenizer: {:?}", tokenizer_path);
    println!("📁 Config: {:?}", config_path);
    println!("📁 Model: {:?}", model_path);

    //Load model config
    let config_str = std::fs::read_to_string(config_path).unwrap();
    let config: Value = serde_json::from_str(&config_str).unwrap();

    println!("\n=== Model Config ===");
    let vocab_size = config.get("vocab_size").unwrap().as_u64().unwrap();
    let num_attention_heads = config.get("num_attention_heads").unwrap().as_u64().unwrap();
    let hidden_size = config.get("hidden_size").unwrap().as_u64().unwrap();
    let num_key_value_heads = config.get("num_key_value_heads").unwrap().as_u64().unwrap();
    let num_layers = config
        .get("num_hidden_layers")
        .or_else(|| config.get("num_layers"))
        .or_else(|| config.get("n_layer"))
        .or_else(|| config.get("n_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let head_dim = hidden_size / num_attention_heads;
    let torch_dtype = config.get("torch_dtype").unwrap().as_str().unwrap();
    let size_of_dtype = get_bytes_per_element(torch_dtype);
    let bytes_per_token = 2 * num_layers * num_key_value_heads * head_dim * size_of_dtype;
    let kib_per_token = bytes_per_token as f64 / 1000.0;

    println!("Vocab Size: {}", vocab_size);
    println!("Hidden Size: {}", hidden_size);
    println!("Number of Layers: {}", num_layers);
    println!("Number of Key-Value Heads: {}", num_key_value_heads);
    println!("Number of Attention Heads: {}", num_attention_heads);
    println!("Head Dimension: {}", head_dim);
    println!("Size of Token: {} KiB", kib_per_token);
    println!("====================\n");

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Encode some text
    let text = "Hey there! This is a mini inference engine.";
    let encoded_prompt = tokenizer.encode(text, false).unwrap();
    let token_ids = encoded_prompt.get_ids();

    println!("Original text: {}", text);
    println!("Tokens: {:?}", encoded_prompt.get_tokens());
    println!("Token IDs: {:?}", token_ids);
}

fn get_bytes_per_element(torch_dtype: &str) -> u64 {
    // Handle different torch_dtype formats
    // Examples: "torch.float16", "float16", "torch.float32", "float32", etc.

    // Extract just the numeric part
    let numeric_part = torch_dtype
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>();

    let parsed_num = numeric_part.parse::<u64>().unwrap();

    parsed_num / 8
}
