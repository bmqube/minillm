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
}
