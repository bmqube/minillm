use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

use super::block::Qwen3Block;
use super::config::Qwen3Config;
use crate::dtype::Precision;
use crate::kv_cache::{KvCache, LayerKind};
use crate::layers::{mask, RotaryEmbedding};
use crate::models::{check_cache, last_position_logits, CausalLM, ModelMeta};

pub struct Qwen3Model {
    cfg: Qwen3Config,
    meta: ModelMeta,
    device: Device,
    precision: Precision,
    embed_tokens: Embedding,
    blocks: Vec<Qwen3Block>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    /// The sliding-window span, when at least one layer uses it.
    window: Option<usize>,
    /// Per layer: whether it attends through the window rather than globally.
    sliding: Vec<bool>,
}

impl Qwen3Model {
    /// The configuration this model was built from.
    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    /// Build from a `Qwen3ForCausalLM` checkpoint.
    ///
    /// `vb` must be rooted at the checkpoint's top level; the transformer body
    /// lives under `model.` and an untied LM head at `lm_head.`.
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let precision = Precision::from_dtype(vb.dtype()).ok_or_else(|| {
            candle_core::Error::Msg(format!("unsupported model dtype {:?}", vb.dtype()))
        })?;

        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(candle_core::Error::Msg(format!(
                "num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }
        if let Some(scaling) = &cfg.rope_scaling {
            return Err(candle_core::Error::Msg(format!(
                "unsupported rope_scaling {scaling}: only plain RoPE is implemented"
            )));
        }

        let model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let layers = model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            blocks.push(Qwen3Block::new(cfg, layers.pp(i.to_string()))?);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?;

        // Small Qwen3 checkpoints tie the LM head to the embedding table; larger
        // ones ship a separate `lm_head.weight`. Trust the tensor's presence over
        // the config flag — a mismatch between the two is a real checkpoint
        // problem and should surface as a shape error, not a silent wrong head.
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        // One table covering every position the model can address:
        // `max_position_embeddings * head_dim / 2` elements each of cos and sin,
        // paid once at load. For Qwen3's 40960-position window and head_dim 128
        // that is ~10 MiB at bf16, ~21 MiB at fp32. A 128k-context checkpoint
        // would want a cap here; at 40k it is not worth the extra knob.
        let rope = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
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
            window: (0..cfg.num_hidden_layers).find_map(|i| cfg.sliding_window_for(i)),
            sliding: (0..cfg.num_hidden_layers)
                .map(|i| cfg.sliding_window_for(i).is_some())
                .collect(),
        })
    }

    /// The step's additive attention masks, `(global, windowed)`; `None` where
    /// none is needed.
    ///
    /// A single new query under full causal attention is the newest position and
    /// may attend to everything cached, so it needs no mask — the common decode
    /// case skips the allocation and the `broadcast_add` entirely. With a sliding
    /// window even a single query needs a mask, because it must *not* see the
    /// oldest cached keys.
    fn build_masks(
        &self,
        seq_len: usize,
        total: usize,
        past: usize,
        dtype: candle_core::DType,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let global = if seq_len == 1 {
            None
        } else {
            Some(mask::causal(seq_len, total, past, dtype, &self.device)?)
        };
        let windowed = self
            .window
            .map(|w| mask::causal_sliding(seq_len, total, past, w, dtype, &self.device))
            .transpose()?;
        Ok((global, windowed))
    }

    /// Embeddings through every block and the final RMSNorm — everything
    /// [`forward`](CausalLM::forward) does except the `lm_head` projection.
    fn hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        if seq_len > self.cfg.max_position_embeddings {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {seq_len} exceeds the model context window {}",
                self.cfg.max_position_embeddings
            )));
        }

        let mut hidden = input_ids.apply(&self.embed_tokens)?;
        let (global, windowed) = self.build_masks(seq_len, seq_len, 0, hidden.dtype())?;
        for (block, &sliding) in self.blocks.iter().zip(&self.sliding) {
            let m = if sliding { &windowed } else { &global };
            hidden = block.forward(&hidden, m.as_ref(), &self.rope)?;
        }
        hidden.apply(&self.norm)
    }

    /// Cache-aware counterpart of [`hidden_states`](Self::hidden_states).
    ///
    /// Note that a sliding window here only *masks* old keys; they stay in the
    /// cache. Evicting them would shrink the footprint but changes what
    /// `cache.len()` means, so it is left to a future cache that owns eviction.
    fn hidden_states_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let (past, total) = check_cache(
            cache,
            self.blocks.len(),
            |_| LayerKind::Attention,
            seq_len,
            self.cfg.max_position_embeddings,
        )?;

        let mut hidden = input_ids.apply(&self.embed_tokens)?;
        let (global, windowed) = self.build_masks(seq_len, total, past, hidden.dtype())?;
        for (i, block) in self.blocks.iter().enumerate() {
            let m = if self.sliding[i] { &windowed } else { &global };
            hidden =
                block.forward_with_cache(&hidden, m.as_ref(), cache.layer(i), past, &self.rope)?;
        }
        hidden.apply(&self.norm)
    }
}

impl CausalLM for Qwen3Model {
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
        last_position_logits(&self.hidden_states(input_ids)?, &self.lm_head)
    }

    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        self.hidden_states_with_cache(input_ids, cache)?
            .apply(&self.lm_head)
    }

    fn forward_with_cache_last(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.hidden_states_with_cache(input_ids, cache)?;
        last_position_logits(&hidden, &self.lm_head)
    }
}
