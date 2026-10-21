//! Qwen3.5 (`Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`, dense text model).
//!
//! The first hybrid architecture in this crate. Three of every four layers
//! replace softmax attention with a **Gated DeltaNet**, a linear-attention
//! mixer whose decode state is a fixed-size matrix per head instead of a KV
//! cache that grows with the sequence. Every fourth layer keeps full attention.
//! The same `qwen3_5_text` architecture also ships as Qwen3.6-27B and
//! Qwen3.8-27B.
//!
//! | | Qwen3 | Qwen3.5 |
//! |---|---|---|
//! | token mixer | GQA attention in every layer | Gated DeltaNet x3, gated GQA attention x1 |
//! | decode state | KV cache per layer | KV cache in 1 layer of 4, fixed recurrent state in the rest |
//! | attention output | `o_proj(attn)` | `o_proj(attn · sigmoid(gate))`, gate from a 2x-wide `q_proj` |
//! | RoPE | whole head, `theta = 1e6` | first 25% of each head, `theta = 1e7` |
//! | RMSNorm gain | `x̂ · w` | `x̂ · (1 + w)`, `w` stored zero-centred |
//! | checkpoint | text only | text under `model.language_model.`, plus vision tower and MTP head |
//!
//! On Qwen3.5-0.8B the KV cache is ~9x smaller per token than Qwen3-0.6B's
//! (6 attention layers x 2 KV heads, against 28 x 8), and the 18 DeltaNet layers
//! add ~19 MB of fp32 state that does not grow with context.
//!
//! # Not implemented
//!
//! - The vision tower and the multi-token-prediction (`mtp.*`) head are in the
//!   checkpoint but not loaded. MTP is a natural drafter for speculative
//!   decoding, later.
//! - Prefill runs the DeltaNet recurrence token by token. The reference's
//!   chunked form is equal in exact arithmetic and faster on long prompts.
//! - The MoE variant (`qwen3_5_moe`, the 35B-A3B models) is rejected at load.

mod attention;
mod block;
mod config;
mod delta_net;
mod model;
mod norm;

pub use attention::Qwen35Attention;
pub use block::Qwen35Block;
pub use config::{Qwen35Config, RopeParameters};
pub use delta_net::GatedDeltaNet;
pub use model::Qwen35Model;
pub use norm::{GatedRmsNorm, ZeroCenteredRmsNorm};
