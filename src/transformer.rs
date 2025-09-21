use crate::tensor::Tensor;
use crate::weights::ModelWeights;
use crate::attention::MultiHeadAttention;
use crate::mlp::MLP;

pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub mlp: MLP,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, n_heads: usize, intermediate_size: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(hidden_size, n_heads),
            mlp: MLP::new(hidden_size, intermediate_size),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        weights: &ModelWeights,
        layer_idx: usize,
    ) -> Tensor {
        // 1. Pre-layer norm + attention + residual connection
        let ln1_weight = &weights.tensors[&format!("h.{}.ln_1.weight", layer_idx)];
        let ln1_bias = &weights.tensors[&format!("h.{}.ln_1.bias", layer_idx)];
        
        let normed_x = x.layer_norm(ln1_weight, ln1_bias, 1e-5);
        let attn_output = self.attention.forward(&normed_x, weights, layer_idx, None);
        let x_after_attn = x.add(&attn_output); // Residual connection
        
        // 2. Pre-layer norm + MLP + residual connection
        let ln2_weight = &weights.tensors[&format!("h.{}.ln_2.weight", layer_idx)];
        let ln2_bias = &weights.tensors[&format!("h.{}.ln_2.bias", layer_idx)];
        
        let normed_x2 = x_after_attn.layer_norm(ln2_weight, ln2_bias, 1e-5);
        let mlp_output = self.mlp.forward(&normed_x2, weights, layer_idx);
        let final_output = x_after_attn.add(&mlp_output); // Residual connection
        
        final_output
    }
}
