//! Token sampling and the autoregressive decode loop.
//!
//! Sampling is done on the CPU over a plain `Vec<f32>`: picking one token per
//! step is cheap, and doing it in scalar Rust keeps the logic obviously correct
//! (the previous tensor-based `nucleus_sampling` returned tokens in sorted order
//! without mapping back to vocabulary ids).
//!
//! [`Generator`] drives the KV-cache decode loop — prefill the prompt once, then
//! feed one token per step. Use [`sample_with`] (and
//! [`Generator::next_token_with`]) with a seeded RNG when a run has to be
//! reproducible; the plain [`sample`] draws from the thread RNG. Greedy decoding
//! (`temperature <= 1e-6`) consumes no randomness at all, so every benchmark in
//! this crate is deterministic regardless.

use candle_core::{Device, IndexOp, Result, Tensor};
use rand::{Rng, RngExt};
use tokenizers::Tokenizer;

use crate::kv_cache::{KvCache, KvQuant};
use crate::models::CausalLM;

/// How to turn a logits vector into the next token id.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 1e-6` means greedy (argmax).
    pub temperature: f64,
    /// Keep only the `k` highest-scoring tokens. `None` or `Some(0)` disables it.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep the shortest ranked prefix whose cumulative
    /// probability mass reaches `p`. `None` disables it.
    pub top_p: Option<f64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

/// Sample the next token id from a logits tensor, drawing from the thread RNG.
///
/// Accepts logits shaped `[vocab]`, `[batch, vocab]` or `[batch, seq, vocab]`.
/// For the batched shapes it uses batch index 0 and the final position.
///
/// Greedy configurations consume no randomness; for a reproducible *sampled* run
/// use [`sample_with`] with a seeded RNG.
pub fn sample(logits: &Tensor, cfg: &SamplingConfig) -> Result<u32> {
    sample_with(logits, cfg, &mut rand::rng())
}

/// [`sample`], but drawing from `rng` — pass a seeded RNG
/// (e.g. `rand::rngs::StdRng::seed_from_u64(0)`) for a reproducible run.
pub fn sample_with<R: Rng + ?Sized>(
    logits: &Tensor,
    cfg: &SamplingConfig,
    rng: &mut R,
) -> Result<u32> {
    // Greedy: top-k and top-p both keep the highest-ranked token, so neither
    // can change the argmax. Take it on the logits' own device.
    if cfg.temperature <= 1e-6 {
        return greedy_id(logits);
    }
    let mut logits = last_step_logits(logits)?;
    if logits.iter().any(|l| l.is_nan()) {
        return Err(candle_core::Error::Msg(
            "sample: logits contain NaN (usually an fp16 overflow; try bf16 or f32)".into(),
        ));
    }

    // Temperature.
    if (cfg.temperature - 1.0).abs() > f64::EPSILON {
        let t = cfg.temperature as f32;
        for l in logits.iter_mut() {
            *l /= t;
        }
    }

    // Rank tokens by descending logit; top-k / top-p slice this order.
    let desc = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]);
    let mut ranked: Vec<usize> = (0..logits.len()).collect();

    // top-k: partition out the k best first, so only those get sorted rather
    // than the whole vocabulary (152k entries on Qwen3) every step.
    if let Some(k) = cfg.top_k.filter(|&k| k > 0 && k < ranked.len()) {
        ranked.select_nth_unstable_by(k - 1, desc);
        ranked.truncate(k);
    }
    ranked.sort_unstable_by(desc);

    // Numerically stable softmax over the kept logits; `ranked[0]` is the max.
    let max_l = logits[ranked[0]];
    let mut probs: Vec<f32> = ranked.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    normalize(&mut probs);

    // top-p (nucleus): truncate to the shortest prefix reaching mass p.
    if let Some(p) = cfg.top_p {
        let p = (p as f32).clamp(0.0, 1.0);
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (rank, &pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p {
                cutoff = rank + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        normalize(&mut probs);
    }

    // Sample from the categorical distribution over the ranked survivors.
    let r: f32 = rng.random::<f32>();
    let mut cum = 0.0f32;
    for (rank, &pr) in probs.iter().enumerate() {
        cum += pr;
        if r <= cum {
            return Ok(ranked[rank] as u32);
        }
    }
    Ok(ranked[probs.len().saturating_sub(1)] as u32)
}

/// The final position's logits row (batch index 0), still on its device.
fn last_step_row(logits: &Tensor) -> Result<Tensor> {
    match logits.dims().len() {
        1 => Ok(logits.clone()),
        2 => logits.i(0),
        3 => match logits.dims()[1] {
            0 => Err(candle_core::Error::Msg(
                "sample: logits have no positions".into(),
            )),
            seq_len => logits.i((0, seq_len - 1)),
        },
        n => Err(candle_core::Error::Msg(format!(
            "sample: expected logits of rank 1-3, got rank {n}"
        ))),
    }
}

/// Copy the logits for the final position into a `Vec<f32>`.
fn last_step_logits(logits: &Tensor) -> Result<Vec<f32>> {
    last_step_row(logits)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()
}

/// Argmax of the final position's logits.
///
/// On a GPU the reduction runs there and only the 4-byte id crosses back to the
/// host — instead of the whole vocabulary row (Qwen3's 151,936 logits are
/// ~300 KB at bf16). On the CPU the row is scanned directly. Exact ties between maximal
/// logits are vanishingly rare, but are not guaranteed to break the same way on
/// both paths.
fn greedy_id(logits: &Tensor) -> Result<u32> {
    let row = last_step_row(logits)?;
    if row.device().is_cpu() {
        Ok(argmax(&last_step_logits(&row)?) as u32)
    } else {
        row.argmax(0)?.to_scalar::<u32>()
    }
}

fn normalize(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Turns generated token ids into printable text, one call per token.
///
/// Byte-level BPE (GPT-2, Qwen) can split one character across several tokens,
/// so decoding token by token prints `�` for each piece. This decodes the whole
/// completion so far and hands back only the new text, holding it back while it
/// still ends in an incomplete character.
pub struct TextStream<'t> {
    tokenizer: &'t Tokenizer,
    ids: Vec<u32>,
    printed: usize,
}

impl<'t> TextStream<'t> {
    pub fn new(tokenizer: &'t Tokenizer) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            printed: 0,
        }
    }

    /// Add `id` and return the text it completes; empty while a character is
    /// still incomplete.
    pub fn push(&mut self, id: u32) -> tokenizers::Result<String> {
        self.ids.push(id);
        let text = self.tokenizer.decode(&self.ids, false)?;
        if text.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        let new = text.get(self.printed..).unwrap_or_default().to_string();
        self.printed = text.len();
        Ok(new)
    }

    /// Every id pushed so far.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// Greedy sampling: pure argmax, no randomness.
pub const GREEDY: SamplingConfig = SamplingConfig {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

/// Drives the KV-cache decode loop: prefill the prompt once, then feed one token
/// per step. Holds the cache and the logits for the most recent position.
///
/// Works with any [`CausalLM`], so the same loop drives GPT-2 and Qwen3
/// unchanged; the device and layer count come from the model.
///
/// ```no_run
/// # use minillm::{device, loader};
/// # use minillm::generation::{Generator, TextStream, GREEDY};
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let dev = device::best();
/// let (model, tok) = loader::load("openai-community/gpt2", &dev)?;
/// let ids = tok.encode("Hello", true)?.get_ids().to_vec();
///
/// let mut gen = Generator::new(model.as_ref());
/// let mut text = TextStream::new(&tok);
/// gen.prefill(&ids)?;
/// for _ in 0..20 {
///     print!("{}", text.push(gen.next_token(&GREEDY)?)?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct Generator<'m> {
    model: &'m dyn CausalLM,
    device: Device,
    cache: KvCache,
    logits: Option<Tensor>,
}

impl<'m> Generator<'m> {
    /// A generator backed by a KV cache at the model's own precision.
    pub fn new(model: &'m dyn CausalLM) -> Self {
        Self::with_quant(model, KvQuant::None)
    }

    /// A generator whose KV cache stores K/V with `quant`.
    pub fn with_quant(model: &'m dyn CausalLM, quant: KvQuant) -> Self {
        Self {
            device: model.device().clone(),
            cache: model.new_cache_with(quant),
            model,
            logits: None,
        }
    }

    /// The model this generator decodes with.
    pub fn model(&self) -> &'m dyn CausalLM {
        self.model
    }

    /// Run `ids` through the model in one pass, seeding the cache. Returns the
    /// logits for the position after the last prompt token.
    ///
    /// Uses [`CausalLM::forward_with_cache_last`] so the LM head runs on just
    /// that one position instead of the whole prompt.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<&Tensor> {
        if ids.is_empty() {
            return Err(candle_core::Error::Msg(
                "prefill needs at least one token".into(),
            ));
        }
        let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        let out = self
            .model
            .forward_with_cache_last(&input, &mut self.cache)?;
        self.logits = Some(out.i(0)?);
        Ok(self.logits.as_ref().expect("just set"))
    }

    /// Feed one token and return the logits it produces.
    pub fn feed(&mut self, id: u32) -> Result<&Tensor> {
        let step = Tensor::from_vec(vec![id], (1, 1), &self.device)?;
        let out = self.model.forward_with_cache_last(&step, &mut self.cache)?;
        self.logits = Some(out.i(0)?);
        Ok(self.logits.as_ref().expect("just set"))
    }

    /// Logits for the most recent position, if [`prefill`](Self::prefill) or
    /// [`feed`](Self::feed) has run.
    pub fn logits(&self) -> Option<&Tensor> {
        self.logits.as_ref()
    }

    /// Sample the next token from the current logits and feed it back, drawing
    /// from the thread RNG. Greedy configs consume no randomness.
    pub fn next_token(&mut self, cfg: &SamplingConfig) -> Result<u32> {
        self.next_token_with(cfg, &mut rand::rng())
    }

    /// [`next_token`](Self::next_token), drawing from `rng`.
    pub fn next_token_with<R: Rng + ?Sized>(
        &mut self,
        cfg: &SamplingConfig,
        rng: &mut R,
    ) -> Result<u32> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("call prefill() before next_token()".into()))?;
        let id = sample_with(logits, cfg, rng)?;
        self.feed(id)?;
        Ok(id)
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The underlying cache.
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// Clear the cache and the pending logits, ready for a fresh prompt.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.logits = None;
    }
}

/// Greedily generate `steps` tokens after `prompt`, using a KV cache with
/// `quant` storage. Returns the generated ids (not including the prompt).
pub fn greedy(
    model: &dyn CausalLM,
    prompt: &[u32],
    steps: usize,
    quant: KvQuant,
) -> Result<Vec<u32>> {
    let mut gen = Generator::with_quant(model, quant);
    gen.prefill(prompt)?;
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        out.push(gen.next_token(&GREEDY)?);
    }
    Ok(out)
}

/// Greedily generate `steps` tokens with **no** cache: every step re-runs the
/// whole sequence through [`CausalLM::forward_last`]. The O(n^2) reference
/// path that [`greedy`] is measured against — and the oracle the cache is
/// checked for equality with. Uses `forward_last` rather than `forward` since
/// only the final position's logits are ever consumed here.
pub fn greedy_no_cache(model: &dyn CausalLM, prompt: &[u32], steps: usize) -> Result<Vec<u32>> {
    let device = model.device().clone();
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let input = Tensor::from_vec(seq.clone(), (1, seq.len()), &device)?;
        let next = greedy_id(&model.forward_last(&input)?)?;
        seq.push(next);
        out.push(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn logits(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1, 1, v.len()), &Device::Cpu).unwrap()
    }

    #[test]
    fn greedy_picks_argmax() {
        let t = logits(&[0.1, 3.0, 0.2, -1.0]);
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        assert_eq!(sample(&t, &cfg).unwrap(), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let t = logits(&[0.1, 0.2, 5.0, 0.2]);
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
        };
        for _ in 0..25 {
            assert_eq!(sample(&t, &cfg).unwrap(), 2);
        }
    }

    #[test]
    fn greedy_ignores_top_k_and_top_p() {
        let t = logits(&[0.1, 0.2, 5.0, 0.2]);
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: Some(3),
            top_p: Some(0.5),
        };
        assert_eq!(sample(&t, &cfg).unwrap(), 2);
    }

    #[test]
    fn top_k_only_draws_from_the_k_best() {
        let t = logits(&[0.0, 3.0, -1.0, 2.9, 0.5, 2.8]);
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: Some(3),
            top_p: None,
        };
        for _ in 0..200 {
            let id = sample(&t, &cfg).unwrap();
            assert!([1, 3, 5].contains(&id), "drew {id}, outside the top 3");
        }
    }

    #[test]
    fn nan_logits_are_an_error_not_a_silent_token() {
        let t = logits(&[0.5, f32::NAN, 1.0, f32::NAN]);
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: None,
            top_p: None,
        };
        let err = sample(&t, &cfg).unwrap_err().to_string();
        assert!(err.contains("NaN"), "{err}");
    }

    #[test]
    fn text_stream_holds_back_a_split_character() {
        // Byte-level vocabulary for the three UTF-8 bytes of '’' (E2 80 99),
        // one token each, plus "a".
        let json = r#"{
          "version": "1.0", "truncation": null, "padding": null,
          "added_tokens": [], "normalizer": null, "pre_tokenizer": null,
          "post_processor": null,
          "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                      "trim_offsets": true, "use_regex": true},
          "model": {"type": "WordLevel", "unk_token": "a",
                    "vocab": {"a": 0, "\u00e2": 1, "\u0122": 2, "\u013b": 3}}
        }"#;
        let tok: Tokenizer = json.parse().unwrap();
        let mut s = TextStream::new(&tok);
        assert_eq!(s.push(0).unwrap(), "a");
        assert_eq!(s.push(1).unwrap(), "");
        assert_eq!(s.push(2).unwrap(), "");
        assert_eq!(s.push(3).unwrap(), "\u{2019}");
        assert_eq!(s.push(0).unwrap(), "a");
        assert_eq!(s.ids(), &[0, 1, 2, 3, 0]);
    }

    #[test]
    fn nucleus_keeps_dominant_token() {
        // token 0 carries ~0.997 of the mass; p = 0.9 must always select it.
        let t = logits(&[6.0, 0.0, 0.0, 0.0]);
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
        };
        for _ in 0..50 {
            assert_eq!(sample(&t, &cfg).unwrap(), 0);
        }
    }

    #[test]
    fn samples_stay_in_vocab_range() {
        let t = logits(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let cfg = SamplingConfig::default();
        for _ in 0..200 {
            assert!(sample(&t, &cfg).unwrap() < 5);
        }
    }

    #[test]
    fn accepts_rank1_and_rank2() {
        let r1 = Tensor::from_vec(vec![0.0f32, 9.0, 0.0], 3, &Device::Cpu).unwrap();
        let r2 = Tensor::from_vec(vec![0.0f32, 0.0, 9.0], (1, 3), &Device::Cpu).unwrap();
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };
        assert_eq!(sample(&r1, &cfg).unwrap(), 1);
        assert_eq!(sample(&r2, &cfg).unwrap(), 2);
    }
}
