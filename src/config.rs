use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Essential core architecture
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub num_layers: u64,
    pub num_attention_heads: u64,
    pub head_dim: u64,

    // Modern architecture support
    pub num_key_value_heads: u64,

    // Context and positioning
    pub max_position_embeddings: u64,

    // Data types and precision
    pub torch_dtype: String,

    // Activation and normalization
    pub activation_function: String,
    pub layer_norm_epsilon: f64,

    // Optional training parameters (for completeness)
    pub initializer_range: Option<f64>,
    pub attention_dropout: Option<f64>,
    pub embedding_dropout: Option<f64>,
    pub residual_dropout: Option<f64>,

    // Computed values
    pub bytes_per_token: u64,
    pub kib_per_token: f64,

    // Model identification
    pub model_type: String,
    pub architectures: Vec<String>,
}

impl ModelConfig {
    /// Parse model config from HuggingFace JSON format
    pub fn from_hf_config(config: &Value) -> Result<Self, Box<dyn std::error::Error>> {
        // Detect model architecture
        let model_type = detect_model_architecture(config);
        let architectures = extract_architectures(config);

        // Essential core architecture
        let vocab_size = get_config_value_u64(config, &["vocab_size", "vocabulary_size"])
            .ok_or("vocab_size not found")?;

        let hidden_size = get_config_value_u64(
            config,
            &[
                "hidden_size", // BERT, RoBERTa, T5, LLaMA
                "n_embd",      // GPT-2, GPT-Neo
                "d_model",     // T5, some transformers
                "model_dim",   // Some other architectures
            ],
        )
        .ok_or("hidden_size not found")?;

        let num_attention_heads = get_config_value_u64(
            config,
            &[
                "num_attention_heads", // BERT, RoBERTa, T5, LLaMA
                "n_head",              // GPT-2, GPT-Neo
                "num_heads",           // Some architectures
                "attention_heads",     // Alternative naming
            ],
        )
        .ok_or("num_attention_heads not found")?;

        let num_layers = get_config_value_u64(
            config,
            &[
                "num_hidden_layers", // BERT, RoBERTa, LLaMA
                "n_layer",           // GPT-2, GPT-Neo
                "num_layers",        // T5, some others
                "n_layers",          // Alternative
                "depth",             // Some architectures
            ],
        )
        .ok_or("num_layers not found")?;

        let head_dim = hidden_size / num_attention_heads;

        // Modern architecture support
        let num_key_value_heads = get_config_value_u64(
            config,
            &[
                "num_key_value_heads", // LLaMA, newer models
                "num_kv_heads",        // Alternative naming
            ],
        )
        .unwrap_or(1); // Default to same as attention heads

        // Context and positioning
        let max_position_embeddings = get_config_value_u64(
            config,
            &[
                "max_position_embeddings", // BERT, RoBERTa, LLaMA
                "n_positions",             // GPT-2, GPT-Neo
                "max_seq_length",          // Some architectures
                "seq_length",              // Alternative
            ],
        )
        .ok_or("max_position_embeddings not found")?;

        // Data types and precision
        let torch_dtype = get_config_value_str(config, &["torch_dtype", "dtype", "precision"])
            .unwrap_or("float32")
            .to_string();

        // Activation and normalization
        let activation_function = get_config_value_str(
            config,
            &[
                "activation_function", // GPT-2
                "hidden_act",          // BERT, RoBERTa
                "feed_forward_proj",   // T5
                "activation",          // Generic
            ],
        )
        .unwrap_or("gelu")
        .to_string();

        let layer_norm_epsilon = get_config_value_f64(
            config,
            &[
                "layer_norm_epsilon", // Most models
                "layer_norm_eps",     // Alternative
                "rms_norm_eps",       // LLaMA
                "norm_epsilon",       // Generic
            ],
        )
        .unwrap_or(1e-5);

        // Optional training parameters
        let initializer_range = get_config_value_f64(
            config,
            &[
                "initializer_range", // Most models
                "init_std",          // Some models
                "weight_init_std",   // Alternative
            ],
        );

        let attention_dropout = get_config_value_f64(
            config,
            &[
                "attn_pdrop",                   // GPT-2
                "attention_dropout",            // Generic
                "attention_probs_dropout_prob", // BERT
            ],
        );

        let embedding_dropout = get_config_value_f64(
            config,
            &[
                "embd_pdrop",          // GPT-2
                "embed_dropout",       // Generic
                "hidden_dropout_prob", // BERT
            ],
        );

        let residual_dropout = get_config_value_f64(
            config,
            &[
                "resid_pdrop",         // GPT-2
                "residual_dropout",    // Generic
                "hidden_dropout_prob", // BERT (reused)
            ],
        );

        // Compute memory usage
        let size_of_dtype = get_bytes_per_element(&torch_dtype);
        let bytes_per_token = 2 * num_layers * num_key_value_heads * head_dim * size_of_dtype;
        let kib_per_token = bytes_per_token as f64 / 1000.0;

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            max_position_embeddings,
            torch_dtype,
            activation_function,
            layer_norm_epsilon,
            initializer_range,
            attention_dropout,
            embedding_dropout,
            residual_dropout,
            bytes_per_token,
            kib_per_token,
            model_type,
            architectures,
        })
    }

    /// Save config to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load config from JSON file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Print a formatted summary of the config
    pub fn print_summary(&self) {
        println!("\n=== Model Configuration Summary ===");
        println!("Model Type: {} {:?}", self.model_type, self.architectures);

        println!("\n🎯 ESSENTIAL:");
        println!("  Vocab Size: {}", self.vocab_size);
        println!("  Hidden Size: {}", self.hidden_size);
        println!("  Number of Layers: {}", self.num_layers);
        println!("  Attention Heads: {}", self.num_attention_heads);
        println!(
            "  Key-Value Heads: {} {}",
            self.num_key_value_heads,
            if self.num_key_value_heads == self.num_attention_heads {
                "(same as attn)"
            } else {
                "(GQA)"
            }
        );
        println!("  Head Dimension: {}", self.head_dim);
        println!(
            "  Max Position Embeddings: {}",
            self.max_position_embeddings
        );
        println!("  Torch Dtype: {}", self.torch_dtype);

        println!("\n🔧 IMPORTANT:");
        println!("  Activation Function: {}", self.activation_function);

        println!("\n⚙️ OPTIONAL:");
        println!("  Layer Norm Epsilon: {}", self.layer_norm_epsilon);
        if let Some(range) = self.initializer_range {
            println!("  Initializer Range: {}", range);
        }
        if let Some(dropout) = self.attention_dropout {
            println!("  Attention Dropout: {}", dropout);
        }
        if let Some(dropout) = self.embedding_dropout {
            println!("  Embedding Dropout: {}", dropout);
        }
        if let Some(dropout) = self.residual_dropout {
            println!("  Residual Dropout: {}", dropout);
        }

        println!("\n💾 MEMORY USAGE:");
        println!("  Size per Token: {:.2} KiB", self.kib_per_token);
        println!("  Bytes per Token: {} bytes", self.bytes_per_token);
        println!("=====================================\n");
    }
}

// Helper functions
fn get_config_value_u64(config: &Value, keys: &[&str]) -> Option<u64> {
    for key in keys {
        if let Some(value) = config.get(key).and_then(|v| v.as_u64()) {
            return Some(value);
        }
    }
    None
}

fn get_config_value_f64(config: &Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(value) = config.get(key).and_then(|v| v.as_f64()) {
            return Some(value);
        }
    }
    None
}

fn get_config_value_str<'a>(config: &'a Value, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(value) = config.get(key).and_then(|v| v.as_str()) {
            return Some(value);
        }
    }
    None
}

fn detect_model_architecture(config: &Value) -> String {
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        return model_type.to_string();
    }

    if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = architectures.first().and_then(|v| v.as_str()) {
            return arch.to_lowercase();
        }
    }

    // Try to infer from available keys
    if config.get("n_embd").is_some() && config.get("n_head").is_some() {
        return "gpt2".to_string();
    } else if config.get("hidden_size").is_some() && config.get("num_attention_heads").is_some() {
        return "bert".to_string();
    }

    "unknown".to_string()
}

fn extract_architectures(config: &Value) -> Vec<String> {
    if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
        return architectures
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
    }
    vec![]
}

fn get_bytes_per_element(torch_dtype: &str) -> u64 {
    // Handle different torch_dtype formats
    // Examples: "torch.float16", "float16", "torch.float32", "float32", etc.

    // Extract just the numeric part
    let numeric_part = torch_dtype
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>();

    if let Ok(parsed_num) = numeric_part.parse::<u64>() {
        parsed_num / 8
    } else {
        // Default to 4 bytes (float32)
        4
    }
}
