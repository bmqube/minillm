use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

// src/loading.rs
use crate::{config::GPT2Config, model::GPT2Model};

pub fn load(
    model_id: &str,
    device: &Device,
) -> Result<(GPT2Model, Tokenizer), Box<dyn std::error::Error + Send + Sync>> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());

    // Download model files
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    // Load config
    let config_str = std::fs::read_to_string(config_path)?;
    let config: GPT2Config = serde_json::from_str(&config_str)?;

    // Load weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

    Ok((GPT2Model::new(&config, vb)?, tokenizer))
}
