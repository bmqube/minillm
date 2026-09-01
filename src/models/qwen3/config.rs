use crate::models::ModelMeta;

/// Qwen3 `config.json`.
///
/// Field names follow the checkpoint, not this crate's GPT-2 vocabulary, so the
/// mapping stays obvious when reading a real config next to it.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Key/value heads. Fewer than `num_attention_heads` under grouped-query
    /// attention — 8 vs 16 on Qwen3-0.6B.
    pub num_key_value_heads: usize,
    /// Width of one attention head.
    ///
    /// Qwen3 sets this **independently** of `hidden_size / num_attention_heads`:
    /// Qwen3-0.6B has `hidden_size = 1024` and 16 heads, but `head_dim = 128`, so
    /// the QKV projections are wider than the residual stream. Deriving it by
    /// division — the reflex from GPT-2 — silently builds the wrong shapes.
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// Qwen3 drops the QKV bias Qwen2 carried; honoured rather than assumed.
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Sliding-window span, when the checkpoint enables windowed attention.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

impl Qwen3Config {
    /// Query heads per key/value head.
    pub fn n_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Effective sliding-window span, or `None` for full causal attention.
    ///
    /// A checkpoint can carry a `sliding_window` value while leaving
    /// `use_sliding_window` false — the released Qwen3 dense models all do — and
    /// in that case attention is global.
    pub fn effective_sliding_window(&self) -> Option<usize> {
        if self.use_sliding_window {
            self.sliding_window.filter(|w| *w > 0)
        } else {
            None
        }
    }

    /// Analytic parameter count.
    ///
    /// Per layer: QKV projections `hidden * (n_head + 2 * n_kv_head) * head_dim`,
    /// the output projection `n_head * head_dim * hidden`, three SwiGLU matrices
    /// `3 * hidden * intermediate`, two RMSNorm gains `2 * hidden`, and the two
    /// per-head QK-norm gains `2 * head_dim`. Plus the embedding table, the final
    /// norm, and an untied LM head when the checkpoint has one.
    pub fn num_parameters(&self) -> usize {
        let h = self.hidden_size;
        let hd = self.head_dim;
        let q = self.num_attention_heads * hd;
        let kv = self.num_key_value_heads * hd;

        let attn_w = h * q + 2 * (h * kv) + q * h;
        let attn_b = if self.attention_bias { q + 2 * kv } else { 0 };
        let mlp = 3 * h * self.intermediate_size;
        let norms = 2 * h + 2 * hd; // input/post-attention RMSNorm + q_norm/k_norm
        let per_layer = attn_w + attn_b + mlp + norms;

        let embeddings = self.vocab_size * h;
        let head = if self.tie_word_embeddings {
            0
        } else {
            self.vocab_size * h
        };
        embeddings + self.num_hidden_layers * per_layer + h + head
    }

    /// Architecture-independent view of this config.
    pub fn meta(&self) -> ModelMeta {
        ModelMeta {
            architecture: "qwen3",
            n_layer: self.num_hidden_layers,
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

    /// The real Qwen/Qwen3-0.6B config.json, trimmed to the keys we read.
    fn qwen3_0_6b_json() -> &'static str {
        r#"{
          "architectures": ["Qwen3ForCausalLM"],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "max_window_layers": 28,
          "model_type": "qwen3",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 1000000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }"#
    }

    fn cfg() -> Qwen3Config {
        serde_json::from_str(qwen3_0_6b_json()).unwrap()
    }

    #[test]
    fn parses_the_real_checkpoint_config() {
        let c = cfg();
        assert_eq!(c.vocab_size, 151_936);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert!(c.tie_word_embeddings);
        assert!(!c.attention_bias);
    }

    #[test]
    fn head_dim_is_not_hidden_size_over_head_count() {
        // The trap this config exists to document: 1024 / 16 = 64, but the real
        // head_dim is 128.
        let c = cfg();
        assert_ne!(c.head_dim, c.hidden_size / c.num_attention_heads);
        assert_eq!(c.head_dim, 128);
    }

    #[test]
    fn reports_grouped_query_attention() {
        let m = cfg().meta();
        assert_eq!(m.n_kv_groups(), 2);
        assert_eq!(cfg().n_kv_groups(), 2);
        assert!(m.n_kv_head < m.n_head);
    }

    #[test]
    fn kv_cache_accounting_uses_kv_heads_not_query_heads() {
        // 2 * 28 layers * 8 kv heads * 128 dim * 2 B (bf16) = 114,688 B/token.
        let m = cfg().meta();
        assert_eq!(m.kv_cache_bytes_per_token(2), 2 * 28 * 8 * 128 * 2);
        assert_eq!(m.kv_cache_bytes_per_token(2), 114_688);
    }

    #[test]
    fn parameter_count_is_about_600m() {
        let n = cfg().num_parameters();
        assert!(
            (550_000_000..750_000_000).contains(&n),
            "expected ~0.6B params, got {n}"
        );
    }

    #[test]
    fn sliding_window_is_off_unless_explicitly_enabled() {
        assert_eq!(cfg().effective_sliding_window(), None);

        let mut c = cfg();
        c.sliding_window = Some(4096);
        // Still off: the flag governs, not the presence of the value.
        assert_eq!(c.effective_sliding_window(), None);

        c.use_sliding_window = true;
        assert_eq!(c.effective_sliding_window(), Some(4096));
    }

    #[test]
    fn optional_keys_fall_back_to_qwen3_defaults() {
        let minimal = r#"{
          "vocab_size": 100, "hidden_size": 8, "intermediate_size": 16,
          "num_hidden_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 4,
          "max_position_embeddings": 128
        }"#;
        let c: Qwen3Config = serde_json::from_str(minimal).unwrap();
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert_eq!(c.rms_norm_eps, 1e-6);
        assert_eq!(c.hidden_act, "silu");
        assert!(!c.attention_bias);
        assert!(!c.tie_word_embeddings);
    }
}
