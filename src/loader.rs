use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{split_id, HFClientSync};
use tokenizers::Tokenizer;

use crate::{config::GPT2Config, model::GPT2Model};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// The three files a GPT-2 checkpoint needs.
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

/// Resolve `config.json`, `tokenizer.json` and `model.safetensors` for an
/// `"owner/name"` id, downloading through the HuggingFace Hub cache if needed.
/// Set `HF_TOKEN` for gated repos.
pub fn resolve(model_id: &str) -> Result<ModelFiles, BoxErr> {
    let client = HFClientSync::new()?;
    let (owner, name) = split_id(model_id);
    let repo = client.model(owner, name);
    let get = |file: &str| repo.download_file().filename(file).send();
    Ok(ModelFiles {
        config: get("config.json")?,
        tokenizer: get("tokenizer.json")?,
        weights: get("model.safetensors")?,
    })
}

/// Point at a local directory holding `config.json`, `tokenizer.json` and
/// `model.safetensors` (no network).
pub fn dir_files(dir: impl AsRef<Path>) -> ModelFiles {
    let dir = dir.as_ref();
    ModelFiles {
        config: dir.join("config.json"),
        tokenizer: dir.join("tokenizer.json"),
        weights: dir.join("model.safetensors"),
    }
}

/// Load from an `"owner/name"` Hub id **or** a local directory. If `spec` is an
/// existing directory it is used as-is; otherwise it is treated as a Hub id.
pub fn load(spec: &str, device: &Device) -> Result<(GPT2Model, Tokenizer), BoxErr> {
    let files = if Path::new(spec).is_dir() {
        dir_files(spec)
    } else {
        resolve(spec)?
    };
    load_files(&files, device)
}

/// Build the model + tokenizer from already-resolved file paths.
pub fn load_files(files: &ModelFiles, device: &Device) -> Result<(GPT2Model, Tokenizer), BoxErr> {
    let tokenizer = Tokenizer::from_file(&files.tokenizer)?;
    let config: GPT2Config = serde_json::from_str(&std::fs::read_to_string(&files.config)?)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, device)? };
    Ok((GPT2Model::new(&config, vb)?, tokenizer))
}
