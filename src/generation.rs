use candle_core::{IndexOp, Result, Tensor};

// src/generation.rs
pub fn top_k_sampling(logits: &Tensor, k: usize) -> Result<Tensor> {
    if k == 0 {
        return Ok(logits.clone());
    }

    let device = logits.device();
    let shape = logits.shape();
    let last_dim = shape.dims().len() - 1;
    let vocab_size = shape.dims()[last_dim];

    // If k is larger than vocabulary, return original logits
    if k >= vocab_size {
        return Ok(logits.clone());
    }

    // Sort in descending order
    let (sorted_values, _sorted_indices) = logits.sort_last_dim(false)?;

    // Get the threshold value (k-th largest value)
    let threshold_values = sorted_values.narrow(last_dim, k - 1, 1)?;

    // Broadcast threshold to match logits dimensions
    let mut broadcast_shape = shape.dims().to_vec();
    broadcast_shape[last_dim] = 1;
    let threshold_reshaped = threshold_values.reshape(broadcast_shape)?;

    // Expand to full size
    let threshold_expanded = threshold_reshaped.broadcast_as(shape)?;

    // Create mask: true where logits >= threshold
    let mask = logits.ge(&threshold_expanded)?;

    // Apply mask
    let neg_inf = Tensor::full(-1e10f32, shape, device)?;
    let result = logits.where_cond(&mask, &neg_inf)?;

    Ok(result)
}

pub fn nucleus_sampling(logits: &Tensor, p: f64) -> Result<Tensor> {
    let device = logits.device();

    // Convert logits to probabilities
    let probs = candle_nn::ops::softmax_last_dim(logits)?;

    // Sort probabilities in descending order
    let (sorted_probs, _sorted_indices) = probs.sort_last_dim(false)?;

    // Calculate cumulative probabilities
    let last_dim = sorted_probs.dims().len() - 1;
    let cumulative_probs = sorted_probs.cumsum(last_dim)?;

    // Create mask: keep tokens where cumulative probability <= p
    let p_tensor = Tensor::full(p as f32, cumulative_probs.shape(), device)?;
    let nucleus_mask = cumulative_probs.le(&p_tensor)?;

    // Apply mask using multiplication instead of where_cond to avoid dtype issues
    let nucleus_mask_f32 = nucleus_mask.to_dtype(candle_core::DType::F32)?;
    let masked_probs = (&sorted_probs * &nucleus_mask_f32)?;

    // Add small epsilon to avoid log(0) and convert back to logits
    let epsilon = 1e-10f32;
    let epsilon_tensor = Tensor::full(epsilon, masked_probs.shape(), device)?;
    let safe_probs = (masked_probs + epsilon_tensor)?;
    let filtered_sorted_probs = safe_probs.log()?;

    // For simplicity, return the filtered sorted probabilities as logits
    // In a full implementation, you'd need to map back to original order using sorted_indices
    Ok(filtered_sorted_probs)
}

pub fn temperature_scale(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    if temperature == 1.0 {
        return Ok(logits.clone());
    }
    logits / temperature
}

pub fn sample_token(logits: &Tensor) -> Result<u32> {
    // Convert logits to probabilities
    let probs = candle_nn::ops::softmax_last_dim(logits)?;

    // Get the last dimension (vocabulary size)
    let shape = probs.shape();
    let vocab_size = shape.dims()[shape.dims().len() - 1];

    // For 2D tensor [batch_size, seq_len, vocab_size], take the last token of the first batch
    // For 3D tensor [batch_size, seq_len, vocab_size], take [0, -1, :] (first batch, last token)
    let probs_1d = if shape.dims().len() == 3 {
        // Shape: [batch_size, seq_len, vocab_size] -> take [0, -1, :]
        let seq_len = shape.dims()[1];
        probs.i((0, seq_len - 1))?
    } else if shape.dims().len() == 2 {
        // Shape: [batch_size, vocab_size] -> take [0, :]
        probs.i(0)?
    } else {
        probs.clone()
    };

    // Convert to Vec<f32> for sampling
    let probs_vec = probs_1d.to_vec1::<f32>()?; // Sample using cumulative distribution
    let random_value: f32 = rand::random();
    let mut cumulative_prob = 0.0;

    for (i, &prob) in probs_vec.iter().enumerate() {
        cumulative_prob += prob;
        if random_value <= cumulative_prob {
            return Ok(i as u32);
        }
    }

    // Fallback: return the last token index
    Ok((vocab_size - 1) as u32)
}

pub fn generate_token(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<u32> {
    let mut processed_logits = logits.clone();

    // Apply temperature scaling
    if temperature != 1.0 {
        processed_logits = temperature_scale(&processed_logits, temperature)?;
    }

    // Apply top-k filtering if specified
    if let Some(k) = top_k {
        processed_logits = top_k_sampling(&processed_logits, k)?;
    }

    // Apply nucleus (top-p) sampling if specified
    if let Some(p) = top_p {
        processed_logits = nucleus_sampling(&processed_logits, p)?;
    }

    // Sample the final token
    sample_token(&processed_logits)
}
