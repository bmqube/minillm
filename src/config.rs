#[derive(Debug, Clone, serde::Deserialize)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
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
}
