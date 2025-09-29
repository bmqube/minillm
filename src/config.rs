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
