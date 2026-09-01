use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use super::block::TransformerBlock;
use super::config::GPT2Config;
use crate::dtype::Precision;
use crate::kv_cache::KvCache;
use crate::layers::mask;
use crate::models::{check_cache, CausalLM, ModelMeta};

pub struct GPT2Model {
    cfg: GPT2Config,
    meta: ModelMeta,
    device: Device,
    precision: Precision,
    wte: candle_nn::Embedding, // Token embeddings
    wpe: candle_nn::Embedding, // Position embeddings
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT2Model {
    /// The configuration this model was built from.
    pub fn config(&self) -> &GPT2Config {
        &self.cfg
    }

    pub fn new(cfg: &GPT2Config, vb: VarBuilder) -> Result<Self> {
        // `layers::gelu` implements exactly one formula (the tanh approximation
        // shared by "gelu_new" and "gelu_pytorch_tanh"); a checkpoint requesting
        // a different activation would silently get the wrong one, so reject it.
        match cfg.activation_function.as_str() {
            "gelu_new" | "gelu_pytorch_tanh" => {}
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unsupported activation_function {other:?}: this crate implements only \
                     the gelu_new / gelu_pytorch_tanh tanh approximation"
                )))
            }
        }

        let device = vb.device().clone();
        let precision = Precision::from_dtype(vb.dtype()).ok_or_else(|| {
            candle_core::Error::Msg(format!("unsupported model dtype {:?}", vb.dtype()))
        })?;

        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(cfg.n_ctx, cfg.n_embd, vb.pp("wpe"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(TransformerBlock::new(cfg, vb.pp(format!("h.{i}")))?);
        }

        let ln_f = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_f"))?;

        // GPT-2 models typically share weights between wte and lm_head. If the
        // checkpoint carries a separate `lm_head.weight`, it's stored `nn.Linear`-style
        // as `(vocab_size, n_embd)` (no transpose needed, unlike the Conv1D-style
        // `c_attn`/`c_proj`/`mlp.*` weights). Only fall back to the tied `wte`
        // weights when the key is genuinely absent; a present-but-wrong-shape key
        // is a real checkpoint mismatch and should error, not silently mis-load.
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            let w = vb.get((cfg.vocab_size, cfg.n_embd), "lm_head.weight")?;
            Linear::new(w, None)
        } else {
            Linear::new(wte.embeddings().clone(), None)
        };

        Ok(Self {
            cfg: cfg.clone(),
            meta: cfg.meta(),
            device,
            precision,
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    /// Token + position embeddings through every block and the final LayerNorm —
    /// everything [`forward`](CausalLM::forward) does except the `lm_head`
    /// projection.
    fn hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        if seq_len > self.cfg.n_ctx {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {seq_len} exceeds GPT-2 context window {}",
                self.cfg.n_ctx
            )));
        }

        let positions = Tensor::arange(0, seq_len as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;

        let tok_emb = input_ids.apply(&self.wte)?;
        let pos_emb = positions.apply(&self.wpe)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        let m = mask::causal(
            seq_len,
            seq_len,
            0,
            hidden_states.dtype(),
            input_ids.device(),
        )?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, Some(&m))?;
        }

        hidden_states.apply(&self.ln_f)
    }

    /// Cache-aware counterpart of [`hidden_states`](Self::hidden_states).
    fn hidden_states_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let (past, total) = check_cache(cache, self.blocks.len(), seq_len, self.cfg.n_ctx)?;

        let positions = Tensor::arange(past as i64, total as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;

        let tok_emb = input_ids.apply(&self.wte)?;
        let pos_emb = positions.apply(&self.wpe)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        // A single new query is the newest position and may attend to
        // everything cached, so it needs no mask at all.
        let m = if seq_len == 1 {
            None
        } else {
            Some(mask::causal(
                seq_len,
                total,
                past,
                hidden_states.dtype(),
                input_ids.device(),
            )?)
        };

        for (i, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward_with_cache(&hidden_states, m.as_ref(), cache.layer(i))?;
        }

        hidden_states.apply(&self.ln_f)
    }
}

impl CausalLM for GPT2Model {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.hidden_states(input_ids)?.apply(&self.lm_head)
    }

    fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.hidden_states(input_ids)?;
        let last = hidden_states.dim(1)? - 1;
        hidden_states.i((.., last, ..))?.apply(&self.lm_head)
    }

    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        self.hidden_states_with_cache(input_ids, cache)?
            .apply(&self.lm_head)
    }

    fn forward_with_cache_last(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let hidden_states = self.hidden_states_with_cache(input_ids, cache)?;
        let last = hidden_states.dim(1)? - 1;
        hidden_states.i((.., last, ..))?.apply(&self.lm_head)
    }
}
