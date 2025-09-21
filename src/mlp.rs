use crate::tensor::Tensor;
use crate::weights::ModelWeights;

pub struct MLP {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    pub fn forward(&self, x: &Tensor, weights: &ModelWeights, layer_idx: usize) -> Tensor {
        // 1. First linear layer (c_fc): [batch, seq_len, hidden] -> [batch, seq_len, intermediate]
        let fc_weight = &weights.tensors[&format!("h.{}.mlp.c_fc.weight", layer_idx)];
        let fc_bias = &weights.tensors[&format!("h.{}.mlp.c_fc.bias", layer_idx)];

        let intermediate = x.matmul(fc_weight).add(fc_bias);

        // 2. GELU activation
        let activated = intermediate.gelu();

        // 3. Second linear layer (c_proj): [batch, seq_len, intermediate] -> [batch, seq_len, hidden]
        let proj_weight = &weights.tensors[&format!("h.{}.mlp.c_proj.weight", layer_idx)];
        let proj_bias = &weights.tensors[&format!("h.{}.mlp.c_proj.bias", layer_idx)];

        activated.matmul(proj_weight).add(proj_bias)
    }
}
