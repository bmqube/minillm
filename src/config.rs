#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    /// LayerNorm epsilon. Defaults to GPT-2's `1e-5` when the checkpoint omits it.
    pub layer_norm_epsilon: f64,
    /// MLP activation. Defaults to GPT-2's `"gelu_new"`; [`GPT2Model::new`](crate::model::GPT2Model::new)
    /// rejects any other value rather than silently running the wrong activation,
    /// since [`crate::activations::gelu`] implements only this one.
    pub activation_function: String,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

fn default_activation_function() -> String {
    "gelu_new".to_string()
}

/// Deserialization shape for `config.json`. `n_ctx` and `n_positions` name the
/// same value under HF's old and new key; real checkpoints (including
/// `openai-community/gpt2`) carry **both**, so this can't be a `#[serde(alias)]`
/// on one field — serde treats the alias and the real name appearing together
/// as a duplicate key. Kept as two independent optional fields instead, and
/// resolved by hand: `n_ctx` wins when both are present, `n_positions` fills in
/// when only it is, and having neither is an error.
#[derive(serde::Deserialize)]
struct RawConfig {
    vocab_size: usize,
    n_ctx: Option<usize>,
    n_positions: Option<usize>,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    layer_norm_epsilon: f64,
    #[serde(default = "default_activation_function")]
    activation_function: String,
}

impl<'de> serde::Deserialize<'de> for GPT2Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawConfig::deserialize(deserializer)?;
        let n_ctx = raw.n_ctx.or(raw.n_positions).ok_or_else(|| {
            serde::de::Error::custom("config.json has neither `n_ctx` nor `n_positions`")
        })?;
        Ok(GPT2Config {
            vocab_size: raw.vocab_size,
            n_ctx,
            n_embd: raw.n_embd,
            n_layer: raw.n_layer,
            n_head: raw.n_head,
            layer_norm_epsilon: raw.layer_norm_epsilon,
            activation_function: raw.activation_function,
        })
    }
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_epsilon: default_layer_norm_epsilon(),
            activation_function: default_activation_function(),
        }
    }
}

impl GPT2Config {
    /// Analytic parameter count for a GPT-2 style model with tied token
    /// embedding / LM head, learned position embeddings and a 4x MLP ratio.
    ///
    /// Per block: `12 * n_embd^2` (attention `c_attn`/`c_proj` + MLP
    /// `c_fc`/`c_proj` weights) plus `13 * n_embd` biases and LayerNorm gains.
    /// For `gpt2` base this returns ~124.4M.
    pub fn num_parameters(&self) -> usize {
        let e = self.n_embd;
        let embeddings = self.vocab_size * e + self.n_ctx * e;
        let per_block = 12 * e * e + 13 * e;
        embeddings + self.n_layer * per_block + 2 * e // + final LayerNorm
    }

    /// Analytic size of the KV cache per sequence position, in bytes, when the
    /// cached keys/values are stored at `dtype_bytes` bytes per element.
    ///
    /// Each layer caches a key tensor and a value tensor, each with `n_embd`
    /// (`= n_head * head_dim`) elements per position, hence the factor of two.
    /// For `gpt2` base at fp32 this is `2 * 12 * 768 * 4` = 73,728 B/token
    /// (72 KiB); a full 1024-token context costs ~72 MiB.
    pub fn kv_cache_bytes_per_token(&self, dtype_bytes: usize) -> usize {
        2 * self.n_layer * self.n_embd * dtype_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_base_is_about_124m_params() {
        let n = GPT2Config::default().num_parameters();
        assert!(
            (124_000_000..125_000_000).contains(&n),
            "expected ~124M params, got {n}"
        );
    }

    #[test]
    fn gpt2_base_kv_cache_is_72kib_per_token_fp32() {
        assert_eq!(
            GPT2Config::default().kv_cache_bytes_per_token(4),
            2 * 12 * 768 * 4
        );
        assert_eq!(GPT2Config::default().kv_cache_bytes_per_token(4), 73_728);
    }

    fn base_json(extra: &str) -> String {
        format!(r#"{{"vocab_size": 50257, "n_embd": 768, "n_layer": 12, "n_head": 12{extra}}}"#)
    }

    #[test]
    fn deserializes_when_both_n_ctx_and_n_positions_are_present() {
        // The real openai-community/gpt2 config.json carries both keys with the
        // same value; a naive `#[serde(alias = "n_positions")]` on `n_ctx`
        // rejects this as a duplicate field.
        let json = base_json(r#", "n_ctx": 1024, "n_positions": 1024"#);
        let cfg: GPT2Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.n_ctx, 1024);
    }

    #[test]
    fn falls_back_to_n_positions_when_n_ctx_is_absent() {
        let json = base_json(r#", "n_positions": 2048"#);
        let cfg: GPT2Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.n_ctx, 2048);
    }

    #[test]
    fn errors_when_neither_n_ctx_nor_n_positions_is_present() {
        let json = base_json("");
        assert!(serde_json::from_str::<GPT2Config>(&json).is_err());
    }

    #[test]
    fn layer_norm_epsilon_and_activation_function_default_to_gpt2_values() {
        let json = base_json(r#", "n_ctx": 1024"#);
        let cfg: GPT2Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.layer_norm_epsilon, 1e-5);
        assert_eq!(cfg.activation_function, "gelu_new");
    }
}
