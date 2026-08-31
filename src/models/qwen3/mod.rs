//! Qwen3 (`Qwen/Qwen3-0.6B` through `Qwen3-14B`, dense variants).
//!
//! A current-generation decoder, and the reason most of [`crate::layers`]
//! exists. Against GPT-2 it changes essentially every component:
//!
//! | | GPT-2 | Qwen3 |
//! |---|---|---|
//! | positions | learned `wpe` table | RoPE, `rope_theta = 1e6` |
//! | normalization | LayerNorm | RMSNorm |
//! | attention | multi-head, fused QKV | grouped-query, separate Q/K/V |
//! | Q/K scaling | none | per-head RMSNorm (QK-norm) before RoPE |
//! | MLP | 4x, tanh-GELU | SwiGLU (`gate`/`up`/`down`) |
//! | weight layout | Conv1D `[in, out]` | `nn.Linear` `[out, in]` |
//! | biases | everywhere | none (`attention_bias: false`) |
//! | context | 1024 | 40960 |
//!
//! Two of those are load-bearing for the KV-cache work this crate measures:
//!
//! - **Grouped-query attention** shrinks the cache by `n_head / n_kv_head`
//!   before any quantization is applied — 2x on Qwen3-0.6B. Cache accounting
//!   therefore keys off `n_kv_head`, and only the attention matmul expands to
//!   the full query head count.
//! - **`head_dim` is an independent config value**, not `hidden_size / n_head`.
//!   Qwen3-0.6B is 1024-wide with 16 heads of 128, so the QKV projections are
//!   wider than the residual stream.

mod attention;
mod block;
mod config;
mod model;

pub use attention::Qwen3Attention;
pub use block::Qwen3Block;
pub use config::Qwen3Config;
pub use model::Qwen3Model;
