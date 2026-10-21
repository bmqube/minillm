use crate::kv_cache::LayerKind;
use crate::models::ModelMeta;

/// Qwen3.5 text-model config.
///
/// The released checkpoints are `Qwen3_5ForConditionalGeneration`: the language
/// model's config sits under `text_config`, next to a vision tower this crate
/// does not load. [`from_json`](Self::from_json) reads either that layout or a
/// text-only one with the same keys at the top level.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,

    // Full-attention layers.
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// 256 on the small models, independent of `hidden_size / heads` as in Qwen3.
    pub head_dim: usize,
    #[serde(default)]
    pub attention_bias: bool,
    /// `q_proj` also emits a per-head gate; the attention output is multiplied
    /// by its sigmoid before `o_proj`.
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,

    // Gated DeltaNet layers.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    /// Width of the short causal depthwise convolution over Q/K/V.
    pub linear_conv_kernel_dim: usize,

    /// `"linear_attention"` or `"full_attention"` per layer. Derived from
    /// `full_attention_interval` when absent.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_interval")]
    pub full_attention_interval: usize,

    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub rope_parameters: RopeParameters,
}

/// The `rope_parameters` block.
///
/// The checkpoint also carries multimodal-RoPE settings (`mrope_section`,
/// `mrope_interleaved`). They assign rotary frequencies to temporal / height /
/// width position axes; for text all three axes hold the same position, so
/// every assignment reduces to plain RoPE and the fields are not read.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    /// Fraction of `head_dim` that is rotated; the rest passes through. 0.25 on
    /// the released models: 64 of 256 dims.
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_true() -> bool {
    true
}

fn default_interval() -> usize {
    4
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_partial_rotary() -> f64 {
    1.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

impl Qwen35Config {
    /// Parse a `config.json`, reading `text_config` when the checkpoint nests
    /// the language model under it. Rejects anything the implementation would
    /// otherwise run wrongly.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        let top_tie = value.get("tie_word_embeddings").and_then(|v| v.as_bool());
        if let Some(text) = value.get_mut("text_config") {
            value = text.take();
        }
        let mut cfg: Self = serde_json::from_value(value)?;
        if let Some(tie) = top_tie {
            cfg.tie_word_embeddings |= tie;
        }
        if cfg.layer_types.is_empty() {
            cfg.layer_types = (0..cfg.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % cfg.full_attention_interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if let Some(t) = self
            .layer_types
            .iter()
            .find(|t| *t != "linear_attention" && *t != "full_attention")
        {
            return Err(format!("unsupported layer type {t:?}"));
        }
        if self.rope_parameters.rope_type != "default" {
            return Err(format!(
                "unsupported rope_type {:?}: only \"default\" is implemented",
                self.rope_parameters.rope_type
            ));
        }
        if self.hidden_act != "silu" {
            return Err(format!("unsupported hidden_act {:?}", self.hidden_act));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            return Err("head counts must divide evenly into their groups".into());
        }
        if !self.rotary_dim().is_multiple_of(2) || self.rotary_dim() > self.head_dim {
            return Err(format!("bad rotary width {}", self.rotary_dim()));
        }
        Ok(())
    }

    /// What each layer keeps between decode steps.
    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        self.layer_types
            .iter()
            .map(|t| {
                if t == "full_attention" {
                    LayerKind::Attention
                } else {
                    LayerKind::Recurrent
                }
            })
            .collect()
    }

    /// Number of full-attention (KV-cached) layers.
    pub fn n_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| *t == "full_attention")
            .count()
    }

    /// Rotated width of each attention head.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }

    /// Q/K/V channels entering the DeltaNet convolution.
    pub fn conv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Analytic parameter count of the text model. Excludes the vision tower
    /// and the multi-token-prediction head, which ship in the same checkpoint
    /// but are not loaded.
    pub fn num_parameters(&self) -> usize {
        let h = self.hidden_size;
        let hd = self.head_dim;
        let q_heads = self.num_attention_heads * hd;
        let kv = self.num_key_value_heads * hd;
        let gate = if self.attn_output_gate { q_heads } else { 0 };
        let bias = if self.attention_bias {
            q_heads + gate + 2 * kv
        } else {
            0
        };
        let attention = h * (q_heads + gate) + 2 * h * kv + q_heads * h + bias + 2 * hd;

        let value_dim = self.linear_num_value_heads * self.linear_value_head_dim;
        let nv = self.linear_num_value_heads;
        let delta_net = h * self.conv_dim()     // in_proj_qkv
            + h * value_dim                     // in_proj_z
            + 2 * h * nv                        // in_proj_b, in_proj_a
            + self.conv_dim() * self.linear_conv_kernel_dim
            + 2 * nv                            // A_log, dt_bias
            + self.linear_value_head_dim        // gated norm
            + value_dim * h; // out_proj

        let per_layer_common = 3 * h * self.intermediate_size + 2 * h;
        let n_attn = self.n_attention_layers();
        let n_linear = self.num_hidden_layers - n_attn;

        let embeddings = self.vocab_size * h;
        let head = if self.tie_word_embeddings {
            0
        } else {
            self.vocab_size * h
        };
        embeddings
            + n_attn * attention
            + n_linear * delta_net
            + self.num_hidden_layers * per_layer_common
            + h
            + head
    }

    /// Architecture-independent view of this config. Head counts and widths
    /// describe the full-attention layers, which are the only ones with a KV
    /// cache.
    pub fn meta(&self) -> ModelMeta {
        ModelMeta {
            architecture: "qwen3_5",
            n_layer: self.num_hidden_layers,
            n_kv_layers: self.n_attention_layers(),
            n_head: self.num_attention_heads,
            n_kv_head: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_size: self.hidden_size,
            n_ctx: self.max_position_embeddings,
            vocab_size: self.vocab_size,
            n_params: self.num_parameters(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real Qwen/Qwen3.5-0.8B config.json, trimmed to the keys read here
    /// plus enough of the outer layout to exercise `text_config` unwrapping.
    fn qwen3_5_0_8b_json() -> String {
        let mut types = Vec::new();
        for i in 0..24 {
            types.push(if (i + 1) % 4 == 0 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            });
        }
        format!(
            r#"{{
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "tie_word_embeddings": true,
              "text_config": {{
                "attention_bias": false,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "layer_types": [{}],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_5_text",
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": true,
                "vocab_size": 248320,
                "rope_parameters": {{
                  "mrope_interleaved": true,
                  "mrope_section": [11, 11, 10],
                  "rope_type": "default",
                  "rope_theta": 10000000,
                  "partial_rotary_factor": 0.25
                }}
              }},
              "vision_config": {{"depth": 12}}
            }}"#,
            types.join(",")
        )
    }

    #[test]
    fn parses_the_nested_text_config() {
        let cfg = Qwen35Config::from_json(&qwen3_5_0_8b_json()).unwrap();
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.head_dim, 256);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rotary_dim(), 64);
        assert_eq!(cfg.conv_dim(), 6144);
        assert_eq!(cfg.rope_parameters.rope_theta, 1e7);
    }

    #[test]
    fn every_fourth_layer_is_full_attention() {
        let cfg = Qwen35Config::from_json(&qwen3_5_0_8b_json()).unwrap();
        let kinds = cfg.layer_kinds();
        assert_eq!(cfg.n_attention_layers(), 6);
        assert_eq!(kinds[0], LayerKind::Recurrent);
        assert_eq!(kinds[3], LayerKind::Attention);
        assert_eq!(kinds[23], LayerKind::Attention);
    }

    #[test]
    fn layer_types_are_derived_from_the_interval_when_absent() {
        let json = qwen3_5_0_8b_json().replace("\"layer_types\"", "\"unused_layer_types\"");
        let cfg = Qwen35Config::from_json(&json).unwrap();
        assert_eq!(cfg.layer_kinds(), {
            let mut v = vec![LayerKind::Recurrent; 24];
            for i in (3..24).step_by(4) {
                v[i] = LayerKind::Attention;
            }
            v
        });
    }

    #[test]
    fn kv_cache_counts_only_attention_layers() {
        // 6 attention layers x 2 KV heads x 256 dim x (K + V) x 4 B.
        let meta = Qwen35Config::from_json(&qwen3_5_0_8b_json())
            .unwrap()
            .meta();
        assert_eq!(meta.n_kv_layers, 6);
        assert_eq!(meta.kv_cache_bytes_per_token(4), 2 * 6 * 2 * 256 * 4);
    }

    #[test]
    fn parameter_count_matches_the_checkpoint() {
        // Sum of every `model.language_model.*` tensor in the released
        // safetensors file; the vision tower (100.6M) and MTP head (20.5M) are
        // excluded, and the LM head is tied.
        let cfg = Qwen35Config::from_json(&qwen3_5_0_8b_json()).unwrap();
        assert_eq!(cfg.num_parameters(), 752_393_024);
    }

    #[test]
    fn rejects_an_unimplemented_rope_type() {
        let json =
            qwen3_5_0_8b_json().replace("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"");
        assert!(Qwen35Config::from_json(&json).is_err());
    }
}
