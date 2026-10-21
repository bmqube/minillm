use candle_core::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use super::block::Qwen35Block;
use super::config::Qwen35Config;
use super::norm::ZeroCenteredRmsNorm;
use crate::dtype::Precision;
use crate::kv_cache::{KvCache, KvQuant};
use crate::layers::{mask, RotaryEmbedding};
use crate::models::{check_cache, CausalLM, ModelMeta};

pub struct Qwen35Model {
    cfg: Qwen35Config,
    meta: ModelMeta,
    device: Device,
    precision: Precision,
    embed_tokens: Embedding,
    blocks: Vec<Qwen35Block>,
    norm: ZeroCenteredRmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
}

impl Qwen35Model {
    /// The configuration this model was built from.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Build from a Qwen3.5 checkpoint rooted at its top level.
    ///
    /// The released checkpoints are multimodal: the language model lives under
    /// `model.language_model.`, beside `model.visual.` and a `mtp.` head, neither
    /// of which is loaded. A text-only checkpoint keeps it under `model.`.
    pub fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let precision = Precision::from_dtype(vb.dtype()).ok_or_else(|| {
            candle_core::Error::Msg(format!("unsupported model dtype {:?}", vb.dtype()))
        })?;

        let body = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model").pp("language_model")
        } else {
            vb.pp("model")
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, body.pp("embed_tokens"))?;
        let layers = body.pp("layers");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Qwen35Block::new(cfg, i, layers.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let norm = ZeroCenteredRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, body.pp("norm"))?;

        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        // Only the rotated slice of each head needs a table: 64 wide on the
        // released models. 262144 positions x 32 frequencies is ~67 MiB of
        // cos + sin at fp32.
        let rope = RotaryEmbedding::new(
            cfg.rotary_dim(),
            cfg.max_position_embeddings,
            cfg.rope_parameters.rope_theta,
            precision.dtype(),
            &device,
        )?;

        Ok(Self {
            cfg: cfg.clone(),
            meta: cfg.meta(),
            device,
            precision,
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
        })
    }

    /// Embeddings through every block and the final norm, for the new tokens,
    /// continuing from `cache`.
    fn hidden_states(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let (past, total) = check_cache(
            cache,
            self.blocks.len(),
            seq_len,
            self.cfg.max_position_embeddings,
        )?;

        let mut hidden = input_ids.apply(&self.embed_tokens)?;
        // Only the attention layers read the mask; a lone new query may see
        // everything cached, so decode needs none.
        let m = if seq_len == 1 {
            None
        } else {
            Some(mask::causal(
                seq_len,
                total,
                past,
                hidden.dtype(),
                &self.device,
            )?)
        };
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, m.as_ref(), cache, i, past, &self.rope)?;
        }
        self.norm.forward(&hidden)
    }

    fn last_position(&self, hidden: Tensor) -> Result<Tensor> {
        let last = hidden.dim(1)? - 1;
        hidden.i((.., last, ..))?.apply(&self.lm_head)
    }
}

impl CausalLM for Qwen35Model {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    /// Full-sequence forward. The DeltaNet layers are recurrent, so there is no
    /// cache-free formulation to keep separately: this runs the cached path on
    /// a fresh cache. `cache_matches_no_cache` then checks that decoding from a
    /// carried state matches recomputing it from scratch every step.
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut cache = self.new_cache();
        self.hidden_states(input_ids, &mut cache)?
            .apply(&self.lm_head)
    }

    fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut cache = self.new_cache();
        let hidden = self.hidden_states(input_ids, &mut cache)?;
        self.last_position(hidden)
    }

    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        self.hidden_states(input_ids, cache)?.apply(&self.lm_head)
    }

    fn forward_with_cache_last(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.hidden_states(input_ids, cache)?;
        self.last_position(hidden)
    }

    fn new_cache_with(&self, quant: KvQuant) -> KvCache {
        KvCache::with_layout(&self.cfg.layer_kinds(), quant)
    }
}
