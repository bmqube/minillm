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
//! (`temperature <= 1e-6`, no `top_k`/`top_p`) consumes no randomness at all, so
//! every benchmark in this crate is deterministic regardless.

use candle_core::{Device, IndexOp, Result, Tensor};
use rand::{Rng, RngExt};

use crate::kv_cache::{KvCache, KvQuant};
use crate::model::GPT2Model;

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
    let mut logits = last_step_logits(logits)?;
    let greedy = cfg.temperature <= 1e-6;

    // Fast path: pure argmax.
    if greedy && cfg.top_k.is_none() && cfg.top_p.is_none() {
        return Ok(argmax(&logits) as u32);
    }

    // Temperature.
    if !greedy && (cfg.temperature - 1.0).abs() > f64::EPSILON {
        let t = cfg.temperature as f32;
        for l in logits.iter_mut() {
            *l /= t;
        }
    }

    // Rank tokens once by descending logit; top-k / top-p slice this order.
    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    ranked.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // top-k: keep at most k candidates.
    let mut keep = ranked.len();
    if let Some(k) = cfg.top_k {
        if k > 0 {
            keep = keep.min(k);
        }
    }
    let candidates = &ranked[..keep];

    // Numerically stable softmax over the kept logits.
    let max_l = candidates
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = candidates
        .iter()
        .map(|&i| (logits[i] - max_l).exp())
        .collect();
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

    // Temperature drove us to greedy but filters are active: argmax the survivors.
    if greedy {
        return Ok(ranked[argmax(&probs)] as u32);
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

/// Backwards-compatible wrapper. Prefer [`sample`] with a [`SamplingConfig`].
pub fn generate_token(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) -> Result<u32> {
    sample(
        logits,
        &SamplingConfig {
            temperature,
            top_k,
            top_p,
        },
    )
}

/// Copy the logits for the final position into a `Vec<f32>`.
fn last_step_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let row = match logits.dims().len() {
        1 => logits.clone(),
        2 => logits.i(0)?,
        3 => {
            let seq_len = logits.dims()[1];
            logits.i((0, seq_len - 1))?
        }
        n => {
            return Err(candle_core::Error::Msg(format!(
                "sample: expected logits of rank 1-3, got rank {n}"
            )))
        }
    };
    row.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()
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

/// Greedy sampling: pure argmax, no randomness.
pub const GREEDY: SamplingConfig = SamplingConfig {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

/// Drives the KV-cache decode loop: prefill the prompt once, then feed one token
/// per step. Holds the cache and the logits for the most recent position.
///
/// ```no_run
/// # use minillm::{device, loader};
/// # use minillm::generation::{Generator, GREEDY};
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let dev = device::best();
/// let (model, tok) = loader::load("openai-community/gpt2", &dev)?;
/// let ids = tok.encode("Hello", true).unwrap().get_ids().to_vec();
///
/// let mut gen = Generator::new(&model, &dev);
/// gen.prefill(&ids)?;
/// for _ in 0..20 {
///     let id = gen.next_token(&GREEDY)?;
///     print!("{}", tok.decode(&[id], false).unwrap());
/// }
/// # Ok(())
/// # }
/// ```
pub struct Generator<'m> {
    model: &'m GPT2Model,
    device: Device,
    cache: KvCache,
    logits: Option<Tensor>,
}

impl<'m> Generator<'m> {
    /// A generator backed by a full-precision KV cache.
    pub fn new(model: &'m GPT2Model, device: &Device) -> Self {
        Self::with_quant(model, device, KvQuant::None)
    }

    /// A generator whose KV cache stores K/V with `quant`.
    pub fn with_quant(model: &'m GPT2Model, device: &Device, quant: KvQuant) -> Self {
        Self {
            model,
            device: device.clone(),
            cache: KvCache::with_quant(model.config().n_layer, quant),
            logits: None,
        }
    }

    /// Run `ids` through the model in one pass, seeding the cache. Returns the
    /// logits for the position after the last prompt token.
    ///
    /// Uses [`GPT2Model::forward_with_cache_last`] so `lm_head` runs on just
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
    model: &GPT2Model,
    device: &Device,
    prompt: &[u32],
    steps: usize,
    quant: KvQuant,
) -> Result<Vec<u32>> {
    let mut gen = Generator::with_quant(model, device, quant);
    gen.prefill(prompt)?;
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        out.push(gen.next_token(&GREEDY)?);
    }
    Ok(out)
}

/// Greedily generate `steps` tokens with **no** cache: every step re-runs the
/// whole sequence through [`GPT2Model::forward_last`]. The O(n^2) reference
/// path that [`greedy`] is measured against — and the oracle the cache is
/// checked for equality with. Uses `forward_last` rather than `forward` since
/// only the final position's logits are ever consumed here.
pub fn greedy_no_cache(
    model: &GPT2Model,
    device: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<u32>> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let input = Tensor::from_vec(seq.clone(), (1, seq.len()), device)?;
        let row: Vec<f32> = model
            .forward_last(&input)?
            .i(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        let next = argmax(&row) as u32;
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
