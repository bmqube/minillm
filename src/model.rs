use crate::config::GPT2Config;
use crate::kv_cache::KvCache;
use crate::transformers::TransformerBlock;
use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

pub struct GPT2Model {
    cfg: GPT2Config,
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
        // `activations::gelu` implements exactly one formula (the tanh
        // approximation shared by "gelu_new" and "gelu_pytorch_tanh"); a
        // checkpoint requesting a different activation would silently get the
        // wrong one, so reject it instead.
        match cfg.activation_function.as_str() {
            "gelu_new" | "gelu_pytorch_tanh" => {}
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unsupported activation_function {other:?}: this crate implements only \
                     the gelu_new / gelu_pytorch_tanh tanh approximation"
                )))
            }
        }

        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(cfg.n_ctx, cfg.n_embd, vb.pp("wpe"))?;

        let mut blocks = Vec::new();
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
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    /// Full-sequence forward pass. Every call recomputes attention over the whole
    /// input; use [`forward_with_cache`](Self::forward_with_cache) for
    /// autoregressive decoding. Runs `lm_head` over **every** position (needed by
    /// callers like perplexity scoring, which score every position) — use
    /// [`forward_last`](Self::forward_last) when only the final position's
    /// logits are needed, e.g. greedy decode without a cache: the `lm_head`
    /// matmul over `[seq, n_embd] x [n_embd, vocab]` dominates prefill-sized
    /// inputs, and its output for every position but the last would otherwise be
    /// computed and immediately discarded.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.hidden_states(input_ids)?;
        hidden_states.apply(&self.lm_head)
    }

    /// [`forward`](Self::forward), but only the final position is projected
    /// through `lm_head`. Returns `[batch, vocab]`. Numerically identical to
    /// `forward(input_ids)` narrowed to its last position, at a fraction of the
    /// `lm_head` cost.
    pub fn forward_last(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.hidden_states(input_ids)?;
        let last = hidden_states.dim(1)? - 1;
        hidden_states.i((.., last, ..))?.apply(&self.lm_head)
    }

    /// Token + position embeddings through every block and the final LayerNorm —
    /// everything [`forward`](Self::forward) does except the `lm_head` projection.
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

        let mask = causal_mask(seq_len, seq_len, 0, input_ids.device())?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, Some(&mask))?;
        }

        hidden_states.apply(&self.ln_f)
    }

    /// Incremental forward pass over `input_ids` (the **new** tokens only),
    /// reading and extending `cache`. Returns logits for those new positions,
    /// `[batch, new_seq, vocab]`.
    ///
    /// Pass the full prompt on the first call (prefill) and one token per step
    /// afterwards. Position ids are offset by the cache length so `wpe` stays
    /// correct; a single-token step needs no attention mask, a multi-token step
    /// gets an offset causal mask. With an empty cache and the full sequence this
    /// computes exactly what [`forward`](Self::forward) does. Use
    /// [`forward_with_cache_last`](Self::forward_with_cache_last) when only the
    /// final position's logits are needed, e.g. the prefill step before a decode
    /// loop.
    pub fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let hidden_states = self.hidden_states_with_cache(input_ids, cache)?;
        hidden_states.apply(&self.lm_head)
    }

    /// [`forward_with_cache`](Self::forward_with_cache), but only the final new
    /// position is projected through `lm_head`. Returns `[batch, vocab]`. On a
    /// single-token step (`new_seq == 1`, the common decode case) this does the
    /// same work as `forward_with_cache`; the saving is on prefill, where
    /// `new_seq` is the whole prompt.
    pub fn forward_with_cache_last(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let hidden_states = self.hidden_states_with_cache(input_ids, cache)?;
        let last = hidden_states.dim(1)? - 1;
        hidden_states.i((.., last, ..))?.apply(&self.lm_head)
    }

    /// Cache-aware counterpart of [`hidden_states`](Self::hidden_states).
    fn hidden_states_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let past = cache.len();
        let total = past + seq_len;
        if total > self.cfg.n_ctx {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {total} exceeds GPT-2 context window {}",
                self.cfg.n_ctx
            )));
        }
        if cache.n_layer() != self.blocks.len() {
            return Err(candle_core::Error::Msg(format!(
                "cache has {} layers, model has {}",
                cache.n_layer(),
                self.blocks.len()
            )));
        }

        let positions = Tensor::arange(past as i64, total as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;

        let tok_emb = input_ids.apply(&self.wte)?;
        let pos_emb = positions.apply(&self.wpe)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(causal_mask(seq_len, total, past, input_ids.device())?)
        };

        for (i, block) in self.blocks.iter().enumerate() {
            hidden_states =
                block.forward_with_cache(&hidden_states, mask.as_ref(), cache.layer(i))?;
        }

        hidden_states.apply(&self.ln_f)
    }
}

/// Additive attention mask, `[q_len, kv_len]`: `0` where a query position may
/// attend to a key position and `-1e10` where it may not.
///
/// Query row `i` is the token at absolute position `past + i` and may attend to
/// key columns `0..=past + i`. With `past == 0` and `q_len == kv_len` this is the
/// plain lower-triangular causal mask.
fn causal_mask(q_len: usize, kv_len: usize, past: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let allowed = past + i; // last key column this query may see
        for j in (allowed + 1)..kv_len {
            data[i * kv_len + j] = -1e10f32;
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), device)
}
