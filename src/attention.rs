use crate::tensor::Tensor;
use crate::weights::ModelWeights;

pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, n_heads: usize) -> Self {
        assert_eq!(
            hidden_size % n_heads,
            0,
            "hidden_size must be divisible by n_heads"
        );

        Self {
            n_heads,
            head_dim: hidden_size / n_heads,
            hidden_size,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        weights: &ModelWeights,
        layer_idx: usize,
        causal_mask: Option<&Tensor>,
    ) -> Tensor {
        let seq_len = x.shape()[1];
        let batch_size = x.shape()[0];

        // 1. Get combined Q, K, V from c_attn (GPT-2 style: one matrix for all Q,K,V)
        let qkv_weight = &weights.tensors[&format!("h.{}.attn.c_attn.weight", layer_idx)];
        let qkv_bias = &weights.tensors[&format!("h.{}.attn.c_attn.bias", layer_idx)];

        // Linear transformation: [batch, seq_len, hidden] -> [batch, seq_len, 3*hidden]
        let qkv = x.matmul(qkv_weight).add(qkv_bias);

        // 2. Split into Q, K, V: [batch, seq_len, 3*hidden] -> 3 x [batch, seq_len, hidden]
        let (q, k, v) = self.split_qkv(&qkv);

        // 3. Reshape for multi-head: [batch, seq_len, hidden] -> [batch, n_heads, seq_len, head_dim]
        let q = self.reshape_for_heads(&q, batch_size, seq_len);
        let k = self.reshape_for_heads(&k, batch_size, seq_len);
        let v = self.reshape_for_heads(&v, batch_size, seq_len);

        // 4. Compute attention scores: Q @ K^T
        let k_transposed = self.transpose_last_two_dims(&k);
        let scores = q.matmul(&k_transposed);

        // 5. Scale by sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scaled_scores = self.scale_tensor(&scores, scale);

        // 6. Apply causal mask (prevent attending to future tokens)
        let masked_scores = if let Some(mask) = causal_mask {
            self.apply_mask(&scaled_scores, mask)
        } else {
            self.create_and_apply_causal_mask(&scaled_scores, seq_len)
        };

        // 7. Softmax to get attention weights
        let attn_weights = masked_scores.softmax(3); // softmax over last dimension (keys)

        // 8. Apply attention to values: attn_weights @ V
        let attn_output = attn_weights.matmul(&v);

        // 9. Reshape back: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let reshaped_output = self.reshape_from_heads(&attn_output, batch_size, seq_len);

        // 10. Final linear projection
        let proj_weight = &weights.tensors[&format!("h.{}.attn.c_proj.weight", layer_idx)];
        let proj_bias = &weights.tensors[&format!("h.{}.attn.c_proj.bias", layer_idx)];

        reshaped_output.matmul(proj_weight).add(proj_bias)
    }

    fn split_qkv(&self, qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
        let shape = qkv.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let total_dim = shape[2];

        assert_eq!(total_dim, 3 * self.hidden_size, "QKV dimension mismatch");

        // Split the last dimension into 3 equal parts
        let mut q_data = Vec::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                for dim in 0..self.hidden_size {
                    // Q: indices 0 to hidden_size-1
                    let q_idx = batch * seq_len * total_dim + seq * total_dim + dim;
                    q_data.push(qkv.data.as_slice().unwrap()[q_idx]);

                    // K: indices hidden_size to 2*hidden_size-1
                    let k_idx =
                        batch * seq_len * total_dim + seq * total_dim + (self.hidden_size + dim);
                    k_data.push(qkv.data.as_slice().unwrap()[k_idx]);

                    // V: indices 2*hidden_size to 3*hidden_size-1
                    let v_idx = batch * seq_len * total_dim
                        + seq * total_dim
                        + (2 * self.hidden_size + dim);
                    v_data.push(qkv.data.as_slice().unwrap()[v_idx]);
                }
            }
        }

        let qkv_shape = [batch_size, seq_len, self.hidden_size];
        (
            Tensor::from_shape(&qkv_shape, q_data),
            Tensor::from_shape(&qkv_shape, k_data),
            Tensor::from_shape(&qkv_shape, v_data),
        )
    }

    fn reshape_for_heads(&self, tensor: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, seq_len, hidden] -> [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        let mut data = Vec::new();
        let src_data = tensor.data.as_slice().unwrap();

        for batch in 0..batch_size {
            for head in 0..self.n_heads {
                for seq in 0..seq_len {
                    for dim in 0..self.head_dim {
                        let src_idx = batch * seq_len * self.hidden_size
                            + seq * self.hidden_size
                            + head * self.head_dim
                            + dim;
                        data.push(src_data[src_idx]);
                    }
                }
            }
        }

        Tensor::from_shape(&[batch_size, self.n_heads, seq_len, self.head_dim], data)
    }

    fn reshape_from_heads(&self, tensor: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let mut data = Vec::new();
        let src_data = tensor.data.as_slice().unwrap();

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                for head in 0..self.n_heads {
                    for dim in 0..self.head_dim {
                        let src_idx = batch * self.n_heads * seq_len * self.head_dim
                            + head * seq_len * self.head_dim
                            + seq * self.head_dim
                            + dim;
                        data.push(src_data[src_idx]);
                    }
                }
            }
        }

        Tensor::from_shape(&[batch_size, seq_len, self.hidden_size], data)
    }

    fn transpose_last_two_dims(&self, tensor: &Tensor) -> Tensor {
        // [batch, n_heads, seq_len, head_dim] -> [batch, n_heads, head_dim, seq_len]
        let shape = tensor.shape();
        let batch_size = shape[0];
        let n_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        let mut data = Vec::new();
        let src_data = tensor.data.as_slice().unwrap();

        for batch in 0..batch_size {
            for head in 0..n_heads {
                for dim in 0..head_dim {
                    for seq in 0..seq_len {
                        let src_idx = batch * n_heads * seq_len * head_dim
                            + head * seq_len * head_dim
                            + seq * head_dim
                            + dim;
                        data.push(src_data[src_idx]);
                    }
                }
            }
        }

        Tensor::from_shape(&[batch_size, n_heads, head_dim, seq_len], data)
    }

    fn scale_tensor(&self, tensor: &Tensor, scale: f32) -> Tensor {
        let scaled_data: Vec<f32> = tensor
            .data
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x * scale)
            .collect();

        Tensor::from_shape(&tensor.shape(), scaled_data)
    }

    fn create_and_apply_causal_mask(&self, scores: &Tensor, seq_len: usize) -> Tensor {
        let shape = scores.shape();
        let mut masked_data = scores.data.as_slice().unwrap().to_vec();

        let batch_size = shape[0];
        let n_heads = shape[1];

        // Apply causal mask: set future positions to -inf
        for batch in 0..batch_size {
            for head in 0..n_heads {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        // j > i means future position
                        let idx = batch * n_heads * seq_len * seq_len
                            + head * seq_len * seq_len
                            + i * seq_len
                            + j;
                        masked_data[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        Tensor::from_shape(&shape, masked_data)
    }

    fn apply_mask(&self, scores: &Tensor, mask: &Tensor) -> Tensor {
        // Apply custom mask (add mask to scores, where mask has -inf for masked positions)
        scores.add(mask)
    }
}
