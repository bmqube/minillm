//! Checkpoint discovery, architecture dispatch and weight loading.
//!
//! [`load`] takes a Hub id or a local directory and returns a boxed
//! [`CausalLM`]: it reads `config.json`, picks the matching implementation from
//! [`crate::models`], resolves however many safetensors shards the checkpoint
//! has, and builds the model at the chosen precision.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_nn::VarBuilder;
use hf_hub::{split_id, HFClientSync};
use tokenizers::Tokenizer;

use crate::dtype::Precision;
use crate::models::{CausalLM, GPT2Model, Qwen3Model};
use crate::models::{gpt2::GPT2Config, qwen3::Qwen3Config};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Name of the shard index a multi-file checkpoint ships instead of a single
/// `model.safetensors`.
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
const SINGLE_SAFETENSORS: &str = "model.safetensors";

/// The files a checkpoint needs.
///
/// `weights` holds one path for a single-file checkpoint and one per shard for a
/// sharded one — anything above ~2B parameters is usually sharded.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Which implementation a checkpoint maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Gpt2,
    Qwen3,
}

impl Architecture {
    /// Short lower-case name, matching [`crate::models::ModelMeta::architecture`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpt2 => "gpt2",
            Self::Qwen3 => "qwen3",
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The handful of `config.json` keys needed before an architecture is chosen.
#[derive(serde::Deserialize)]
struct ConfigProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    /// The precision the checkpoint was published at. Used as the default when
    /// the caller does not pin one.
    #[serde(default)]
    torch_dtype: Option<String>,
}

impl ConfigProbe {
    fn architecture(&self) -> Result<Architecture, BoxErr> {
        if let Some(t) = self.model_type.as_deref() {
            match t {
                "gpt2" => return Ok(Architecture::Gpt2),
                "qwen3" => return Ok(Architecture::Qwen3),
                _ => {}
            }
        }
        // Fall back to the class name for configs that omit `model_type`.
        for a in self.architectures.iter().flatten() {
            if a.starts_with("GPT2") {
                return Ok(Architecture::Gpt2);
            }
            if a.starts_with("Qwen3") {
                return Ok(Architecture::Qwen3);
            }
        }
        Err(format!(
            "unsupported checkpoint: model_type {:?}, architectures {:?}. \
             This crate implements gpt2 and qwen3.",
            self.model_type, self.architectures
        )
        .into())
    }
}

/// Read `config.json` and report which architecture it describes.
pub fn detect_architecture(config: &Path) -> Result<Architecture, BoxErr> {
    probe(config)?.architecture()
}

fn probe(config: &Path) -> Result<ConfigProbe, BoxErr> {
    Ok(serde_json::from_str(&std::fs::read_to_string(config)?)?)
}

/// Resolve a checkpoint's files from the HuggingFace Hub, downloading through
/// the local cache if needed. Set `HF_TOKEN` for gated repos.
///
/// Tries `model.safetensors` first and falls back to the shard index, which is
/// how every checkpoint too large for a single file is published.
pub fn resolve(model_id: &str) -> Result<ModelFiles, BoxErr> {
    let client = HFClientSync::new()?;
    let (owner, name) = split_id(model_id);
    let repo = client.model(owner, name);
    let get = |file: &str| repo.download_file().filename(file).send();

    let config = get("config.json")?;
    let tokenizer = get("tokenizer.json")?;

    let weights = match get(SINGLE_SAFETENSORS) {
        Ok(single) => vec![single],
        Err(single_err) => {
            // No single-file weights: expect a shard index listing the parts.
            let index = get(SAFETENSORS_INDEX).map_err(|index_err| -> BoxErr {
                format!(
                    "{model_id} has neither {SINGLE_SAFETENSORS} ({single_err}) \
                     nor {SAFETENSORS_INDEX} ({index_err})"
                )
                .into()
            })?;
            let mut shards = Vec::new();
            for name in shard_names(&index)? {
                shards.push(get(&name)?);
            }
            shards
        }
    };

    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

/// Point at a local directory holding `config.json`, `tokenizer.json` and either
/// `model.safetensors` or a shard index plus its parts (no network).
pub fn dir_files(dir: impl AsRef<Path>) -> Result<ModelFiles, BoxErr> {
    let dir = dir.as_ref();
    let single = dir.join(SINGLE_SAFETENSORS);
    let index = dir.join(SAFETENSORS_INDEX);

    let weights = if single.is_file() {
        vec![single]
    } else if index.is_file() {
        shard_names(&index)?.into_iter().map(|n| dir.join(n)).collect()
    } else {
        return Err(format!(
            "{} has neither {SINGLE_SAFETENSORS} nor {SAFETENSORS_INDEX}",
            dir.display()
        )
        .into())
    };

    Ok(ModelFiles {
        config: dir.join("config.json"),
        tokenizer: dir.join("tokenizer.json"),
        weights,
    })
}

/// The distinct shard filenames listed in a `model.safetensors.index.json`.
///
/// The index maps every tensor name to the shard holding it, so the same
/// filename appears hundreds of times; a `BTreeSet` deduplicates and keeps the
/// order stable, which matters because the load order ends up in error messages.
fn shard_names(index: &Path) -> Result<Vec<String>, BoxErr> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: std::collections::HashMap<String, String>,
    }
    let parsed: Index = serde_json::from_str(&std::fs::read_to_string(index)?)?;
    if parsed.weight_map.is_empty() {
        return Err(format!("{} lists no tensors", index.display()).into());
    }
    Ok(parsed
        .weight_map
        .into_values()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect())
}

/// Load from an `"owner/name"` Hub id **or** a local directory, at the
/// checkpoint's own published precision.
///
/// If `spec` is an existing directory it is used as-is; otherwise it is treated
/// as a Hub id. Precision comes from the config's `torch_dtype`, falling back to
/// fp32 — so GPT-2 (`"float32"`) keeps loading exactly as it always has, and
/// Qwen3 (`"bfloat16"`) loads at bf16 instead of silently doubling in size.
/// Use [`load_with`] to pin one explicitly.
pub fn load(spec: &str, device: &Device) -> Result<(Box<dyn CausalLM>, Tokenizer), BoxErr> {
    load_with(spec, device, None)
}

/// [`load`], with an explicit precision override.
pub fn load_with(
    spec: &str,
    device: &Device,
    precision: Option<Precision>,
) -> Result<(Box<dyn CausalLM>, Tokenizer), BoxErr> {
    let files = if Path::new(spec).is_dir() {
        dir_files(spec)?
    } else {
        resolve(spec)?
    };
    load_files(&files, device, precision)
}

/// Build the model + tokenizer from already-resolved file paths.
pub fn load_files(
    files: &ModelFiles,
    device: &Device,
    precision: Option<Precision>,
) -> Result<(Box<dyn CausalLM>, Tokenizer), BoxErr> {
    let probe = probe(&files.config)?;
    let architecture = probe.architecture()?;
    let precision = precision
        .or_else(|| probe.torch_dtype.as_deref().and_then(Precision::parse))
        .unwrap_or_default();

    let tokenizer = Tokenizer::from_file(&files.tokenizer)?;
    let json = std::fs::read_to_string(&files.config)?;

    // Safety: `from_mmaped_safetensors` maps the files for the lifetime of the
    // VarBuilder; mutating them on disk while loaded would be undefined. The
    // caller owns paths it just resolved, so nothing else is writing them.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, precision.dtype(), device)? };

    let model: Box<dyn CausalLM> = match architecture {
        Architecture::Gpt2 => {
            let cfg: GPT2Config = serde_json::from_str(&json)?;
            Box::new(GPT2Model::new(&cfg, vb)?)
        }
        Architecture::Qwen3 => {
            let cfg: Qwen3Config = serde_json::from_str(&json)?;
            Box::new(Qwen3Model::new(&cfg, vb)?)
        }
    };
    Ok((model, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_json(s: &str) -> ConfigProbe {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn detects_gpt2_from_model_type() {
        let p = probe_json(r#"{"model_type": "gpt2", "torch_dtype": "float32"}"#);
        assert_eq!(p.architecture().unwrap(), Architecture::Gpt2);
        assert_eq!(
            Precision::parse(p.torch_dtype.as_deref().unwrap()),
            Some(Precision::F32)
        );
    }

    #[test]
    fn detects_qwen3_from_model_type() {
        let p = probe_json(r#"{"model_type": "qwen3", "torch_dtype": "bfloat16"}"#);
        assert_eq!(p.architecture().unwrap(), Architecture::Qwen3);
        assert_eq!(
            Precision::parse(p.torch_dtype.as_deref().unwrap()),
            Some(Precision::BF16)
        );
    }

    #[test]
    fn falls_back_to_the_architectures_class_name() {
        let p = probe_json(r#"{"architectures": ["Qwen3ForCausalLM"]}"#);
        assert_eq!(p.architecture().unwrap(), Architecture::Qwen3);
        let p = probe_json(r#"{"architectures": ["GPT2LMHeadModel"]}"#);
        assert_eq!(p.architecture().unwrap(), Architecture::Gpt2);
    }

    #[test]
    fn rejects_an_architecture_this_crate_does_not_implement() {
        let p = probe_json(r#"{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}"#);
        let err = p.architecture().unwrap_err().to_string();
        assert!(err.contains("unsupported checkpoint"), "{err}");
        assert!(err.contains("llama"), "{err}");
    }

    #[test]
    fn a_config_with_no_identifying_keys_is_an_error() {
        assert!(probe_json(r#"{"vocab_size": 10}"#).architecture().is_err());
    }

    #[test]
    fn shard_index_is_deduplicated_and_sorted() {
        let dir = std::env::temp_dir().join("minillm-shard-index-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(SAFETENSORS_INDEX);
        std::fs::write(
            &path,
            r#"{"metadata": {"total_size": 1},
                "weight_map": {
                  "b.weight": "model-00002-of-00002.safetensors",
                  "a.weight": "model-00001-of-00002.safetensors",
                  "a.bias":   "model-00001-of-00002.safetensors"
                }}"#,
        )
        .unwrap();

        let names = shard_names(&path).unwrap();
        assert_eq!(
            names,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_shard_index_is_an_error() {
        let dir = std::env::temp_dir().join("minillm-empty-index-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(SAFETENSORS_INDEX);
        std::fs::write(&path, r#"{"weight_map": {}}"#).unwrap();
        assert!(shard_names(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dir_files_reports_a_directory_with_no_weights() {
        let dir = std::env::temp_dir().join("minillm-no-weights-test");
        std::fs::create_dir_all(&dir).unwrap();
        let err = dir_files(&dir).unwrap_err().to_string();
        assert!(err.contains(SINGLE_SAFETENSORS), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
