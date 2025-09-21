//! GPT-style transformer model implementation

use crate::tensor::Tensor;
use crate::transformer::TransformerBlock;
use crate::weights::ModelWeights;

/// Complete GPT-style transformer model
///
/// Includes token embeddings, position embeddings, transformer blocks,
/// and language modeling head for next-token prediction.
pub struct GPTModel {
    _n_layers: usize,
    _n_vocab: usize,
    pub n_ctx: usize,
    _hidden_size: usize,
    _n_heads: usize,
    _intermediate_size: usize,
    pub blocks: Vec<TransformerBlock>,
}

impl GPTModel {
    pub fn new(
        n_layers: usize,
        n_vocab: usize,
        n_ctx: usize,
        hidden_size: usize,
        n_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        let blocks = (0..n_layers)
            .map(|_| TransformerBlock::new(hidden_size, n_heads, intermediate_size))
            .collect();

        Self {
            _n_layers: n_layers,
            _n_vocab: n_vocab,
            n_ctx,
            _hidden_size: hidden_size,
            _n_heads: n_heads,
            _intermediate_size: intermediate_size,
            blocks,
        }
    }

    pub fn forward(&self, token_ids: &[u32], weights: &ModelWeights) -> Tensor {
        let _batch_size = 1; // For now, assume batch size of 1
        let seq_len = token_ids.len();

        // 1. Token embedding lookup
        let token_embeds = self.get_token_embeddings(token_ids, weights);

        // 2. Position embedding
        let pos_embeds = self.get_position_embeddings(seq_len, weights);

        // 3. Add token and position embeddings
        let mut x = token_embeds.add(&pos_embeds);

        // 4. Pass through all transformer blocks
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, weights, layer_idx);
        }

        // 5. Final layer norm
        let ln_f_weight = &weights.tensors["ln_f.weight"];
        let ln_f_bias = &weights.tensors["ln_f.bias"];
        let normalized = x.layer_norm(ln_f_weight, ln_f_bias, 1e-5);

        // 6. Language modeling head (project to vocabulary)
        // In GPT-2, this is typically tied to the token embedding weights
        let wte = &weights.tensors["wte.weight"];
        // For matrix multiplication, we need wte to be [hidden_size, vocab_size]
        // but it's stored as [vocab_size, hidden_size], so we need to transpose
        let wte_t = Tensor {
            data: wte.data.t().to_owned(),
        };
        let logits = normalized.matmul(&wte_t);

        logits
    }

    fn get_token_embeddings(&self, token_ids: &[u32], weights: &ModelWeights) -> Tensor {
        let wte = &weights.tensors["wte.weight"];
        let vocab_size = wte.shape()[0];
        let hidden_size = wte.shape()[1];
        let seq_len = token_ids.len();

        let mut embedding_data = Vec::with_capacity(seq_len * hidden_size);
        let wte_data = wte.data.as_slice().unwrap();

        for &token_id in token_ids {
            let token_id = token_id as usize;
            assert!(
                token_id < vocab_size,
                "Token ID {} out of vocabulary range",
                token_id
            );

            // Extract embedding for this token
            let start_idx = token_id * hidden_size;
            let end_idx = start_idx + hidden_size;
            embedding_data.extend_from_slice(&wte_data[start_idx..end_idx]);
        }

        Tensor::from_shape(&[1, seq_len, hidden_size], embedding_data)
    }

    fn get_position_embeddings(&self, seq_len: usize, weights: &ModelWeights) -> Tensor {
        let wpe = &weights.tensors["wpe.weight"];
        let max_pos = wpe.shape()[0];
        let hidden_size = wpe.shape()[1];

        assert!(
            seq_len <= max_pos,
            "Sequence length {} exceeds maximum position {}",
            seq_len,
            max_pos
        );

        let mut pos_embedding_data = Vec::with_capacity(seq_len * hidden_size);
        let wpe_data = wpe.data.as_slice().unwrap();

        for pos in 0..seq_len {
            let start_idx = pos * hidden_size;
            let end_idx = start_idx + hidden_size;
            pos_embedding_data.extend_from_slice(&wpe_data[start_idx..end_idx]);
        }

        Tensor::from_shape(&[1, seq_len, hidden_size], pos_embedding_data)
    }

    pub fn generate_next_token(&self, token_ids: &[u32], weights: &ModelWeights) -> u32 {
        let logits = self.forward(token_ids, weights);

        // Get logits for the last position (next token prediction)
        let seq_len = token_ids.len();
        let vocab_size = logits.shape()[2];
        let last_logits_start = (seq_len - 1) * vocab_size;
        let last_logits_end = last_logits_start + vocab_size;

        let logits_data = logits.data.as_slice().unwrap();
        let last_logits = &logits_data[last_logits_start..last_logits_end];

        // Simple greedy sampling: pick the token with highest probability
        let mut max_idx = 0;
        let mut max_val = last_logits[0];

        for (idx, &val) in last_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        max_idx as u32
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        weights: &ModelWeights,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // Limit context length to avoid memory issues
            let context_tokens = if tokens.len() > self.n_ctx {
                &tokens[tokens.len() - self.n_ctx..]
            } else {
                &tokens
            };

            let next_token = self.generate_next_token(context_tokens, weights);
            tokens.push(next_token);

            // Stop if we hit an end token (you might want to define this)
            // if next_token == END_TOKEN { break; }
        }

        tokens
    }
}
