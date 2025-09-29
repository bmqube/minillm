use crate::config::GPT2Config;
use crate::transformers::TransformerBlock;
use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

pub struct GPT2Model {
    wte: candle_nn::Embedding, // Token embeddings
    wpe: candle_nn::Embedding, // Position embeddings
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT2Model {
    pub fn new(cfg: &GPT2Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(cfg.n_ctx, cfg.n_embd, vb.pp("wpe"))?;

        let mut blocks = Vec::new();
        for i in 0..cfg.n_layer {
            blocks.push(TransformerBlock::new(cfg, vb.pp(&format!("h.{}", i)))?);
        }

        let ln_f = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;

        // GPT-2 models typically share weights between wte and lm_head
        // Try to load lm_head weights, fallback to wte weights if not found
        let lm_head =
            if let Ok(lm_head_weight) = vb.get((cfg.n_embd, cfg.vocab_size), "lm_head.weight") {
                // lm_head exists, transpose it
                let transposed_weight = lm_head_weight.t()?;
                Linear::new(transposed_weight, None)
            } else {
                // lm_head doesn't exist, use wte weights (weight sharing)
                let wte_weight = wte.embeddings().clone();
                Linear::new(wte_weight, None)
            };

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Create position IDs
        let positions = Tensor::arange(0, seq_len as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;

        // Embeddings
        let tok_emb = input_ids.apply(&self.wte)?;
        let pos_emb = positions.apply(&self.wpe)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        // Create causal mask
        let mask = create_causal_mask(seq_len, input_ids.device())?;

        // Transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, Some(&mask))?;
        }

        // Final layer norm and projection
        let hidden_states = hidden_states.apply(&self.ln_f)?;

        hidden_states.apply(&self.lm_head)
    }
}

fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = -1e10f32; // Mask future positions
            }
        }
    }

    Tensor::from_vec(mask_data, (seq_len, seq_len), device)
}
