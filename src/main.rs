use dotenv::dotenv;
use hf_hub::api::sync::ApiBuilder;
use serde_json::Value;
use std::env;
use tokenizers::Tokenizer;

mod attention;
mod config;
mod gpt;
mod mlp;
mod tensor;
mod transformer;
mod weights;

use crate::config::ModelConfig;
use crate::gpt::GPTModel;
use crate::weights::ModelWeights;

fn main() {
    dotenv().ok();

    let api = ApiBuilder::new()
        .with_token(Some(env::var("HF_TOKEN").unwrap()))
        .build()
        .unwrap();

    let repo = api.model("openai-community/gpt2".to_string());

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
    let config_json: Value = serde_json::from_str(&config_str).unwrap();

    let config = ModelConfig::from_hf_config(&config_json).unwrap();
    config.print_summary();

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Encode some text
    let text = "what is the capital of france?";
    let encoded_prompt = tokenizer.encode(text, false).unwrap();
    let token_ids = encoded_prompt.get_ids();

    println!("Original text: {}", text);
    println!("Tokens: {:?}", encoded_prompt.get_tokens());
    println!("Token IDs: {:?}", token_ids);

    let weights = ModelWeights::load_from_safetensors(&model_path).unwrap();

    // Create GPT model
    let model = GPTModel::new(
        config.num_layers as usize,
        config.vocab_size as usize,
        config.max_position_embeddings as usize,
        config.hidden_size as usize,
        config.num_attention_heads as usize,
        config.hidden_size as usize * 4, // GPT-2 uses 4x hidden size for MLP
    );
    println!("✅ GPT model created with {} layers", config.num_layers);

    // Test with simple inference
    println!("\n🚀 Testing inference...");
    let token_ids_vec: Vec<u32> = token_ids.to_vec();
    let generated_tokens = model.generate(&token_ids_vec, 5, &weights);

    // Decode back to text
    let generated_text = tokenizer.decode(&generated_tokens, true).unwrap();
    println!("✨ Generated text: '{}'", generated_text);
}
