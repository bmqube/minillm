//! A small transformer inference engine in Rust, built on [`candle`].
//!
//! Loads decoder-only checkpoints from the HuggingFace Hub and runs
//! autoregressive generation on CPU or CUDA, with a KV cache that can store its
//! keys and values at full precision, per-token int8, or per-token packed int4.
//!
//! # Layout
//!
//! | Module | What lives there |
//! |---|---|
//! | [`models`] | The [`CausalLM`] trait and one module per architecture ([`models::gpt2`], [`models::qwen3`]) |
//! | [`layers`] | Primitives shared between architectures: activations, masks, RoPE, GQA head expansion |
//! | [`kv_cache`] | The decode-time key/value store and its quantization modes |
//! | [`generation`] | Sampling and the [`Generator`](generation::Generator) decode loop |
//! | [`loader`] | Config parsing, architecture dispatch, sharded safetensors |
//! | [`dtype`] | Compute [`Precision`] selection |
//! | [`device`] | CPU / CUDA selection |
//!
//! Everything above [`models`] is architecture-agnostic: it talks to
//! [`CausalLM`] and [`ModelMeta`], never to a concrete model type.
//!
//! # Example
//!
//! ```no_run
//! use minillm::generation::{Generator, SamplingConfig};
//! use minillm::{device, loader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let dev = device::best();
//! let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;
//!
//! let ids = tokenizer.encode("The future of AI is", true)?.get_ids().to_vec();
//! let cfg = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.95) };
//!
//! let mut generator = Generator::new(model.as_ref());
//! generator.prefill(&ids)?;
//! for _ in 0..40 {
//!     let next = generator.next_token(&cfg)?;
//!     print!("{}", tokenizer.decode(&[next], false)?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`candle`]: https://github.com/huggingface/candle

pub mod device;
pub mod dtype;
pub mod generation;
pub mod kv_cache;
pub mod layers;
pub mod loader;
pub mod models;

pub use dtype::Precision;
pub use kv_cache::{KvCache, KvQuant};
pub use models::{CausalLM, ModelMeta};
