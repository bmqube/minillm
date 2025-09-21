//! High-level inference engine for text generation

use crate::{config::ModelConfig, gpt::GPTModel, weights::ModelWeights, Result};
use dotenv::dotenv;
use hf_hub::api::sync::ApiBuilder;
use serde_json::Value;
use std::env;
use tokenizers::Tokenizer;

/// High-level inference engine that handles model loading and text generation
pub struct InferenceEngine {
    model: GPTModel,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    config: ModelConfig,
}

impl InferenceEngine {
    /// Create a new inference engine for the specified model
    pub fn new(model_name: &str) -> Result<Self> {
        Self::new_with_token(model_name, None)
    }

    /// Create a new inference engine with a custom HuggingFace token
    pub fn new_with_token(model_name: &str, token: Option<String>) -> Result<Self> {
        dotenv().ok();

        let token = token.or_else(|| env::var("HF_TOKEN").ok());
        
        let api = ApiBuilder::new()
            .with_token(token)
            .build()?;

        let repo = api.model(model_name.to_string());

        // Download model files
        let tokenizer_path = repo.get("tokenizer.json")?;
        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;

        // Load model config
        let config_str = std::fs::read_to_string(config_path)?;
        let config_json: Value = serde_json::from_str(&config_str)?;
        let config = ModelConfig::from_hf_config(&config_json)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // Load weights
        let weights = ModelWeights::load_from_safetensors(&model_path)
            .map_err(|e| format!("Failed to load weights: {}", e))?;

        // Create model
        let model = GPTModel::new(
            config.num_layers as usize,
            config.vocab_size as usize,
            config.max_position_embeddings as usize,
            config.hidden_size as usize,
            config.num_attention_heads as usize,
            config.hidden_size as usize * 4,
        );

        Ok(Self {
            model,
            weights,
            tokenizer,
            config,
        })
    }

    /// Generate text continuation for the given prompt
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Tokenize input
        let encoded_prompt = self.tokenizer.encode(prompt, false)?;
        let token_ids: Vec<u32> = encoded_prompt.get_ids().to_vec();

        // Generate tokens
        let generated_tokens = self.model.generate(&token_ids, max_tokens, &self.weights);

        // Decode back to text
        let generated_text = self.tokenizer.decode(&generated_tokens, true)?;
        
        Ok(generated_text)
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Tokenize text without generating
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self.tokenizer.encode(text, false)?;
        Ok(encoded.get_ids().to_vec())
    }

    /// Decode tokens back to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }
}
