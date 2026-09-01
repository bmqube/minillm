//! GPT-2 (`openai-community/gpt2` and its medium / large / xl siblings).
//!
//! The 2019 decoder: learned absolute position embeddings, LayerNorm, plain
//! multi-head attention with a fused QKV projection, and a 4x MLP with the tanh
//! approximation to GELU. Weights are stored HuggingFace `Conv1D`-style
//! (`[in, out]`), so the linear layers transpose on load.
//!
//! Kept as the small, fast, fully parity-checked reference architecture — the
//! sanity check every change is validated against before it is trusted on a
//! larger model.

mod attention;
mod block;
mod config;
mod model;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use config::GPT2Config;
pub use model::GPT2Model;
