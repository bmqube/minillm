//! # MiniLLM - A Mini Transformer Inference Engine
//!
//! A lightweight, efficient transformer inference engine written in Rust.
//! Supports GPT-2 style models with multi-head attention, feed-forward networks,
//! and layer normalization.
//!
//! ## Features
//! - Dynamic tensor operations with ndarray
//! - SafeTensors weight loading from HuggingFace
//! - Complete GPT-2 architecture implementation
//! - Text generation with autoregressive sampling
//!
//! ## Example
//! ```rust,no_run
//! use minillm::inference::InferenceEngine;
//! 
//! let engine = InferenceEngine::new("openai-community/gpt2")?;
//! let result = engine.generate("Hello world", 10)?;
//! println!("Generated: {}", result);
//! ```

pub mod attention;
pub mod config;
pub mod gpt;
pub mod inference;
pub mod mlp; 
pub mod tensor;
pub mod transformer;
pub mod weights;

// Re-export main types for convenience
pub use config::ModelConfig;
pub use gpt::GPTModel;
pub use inference::InferenceEngine;
pub use tensor::Tensor;
pub use weights::ModelWeights;

/// Result type used throughout the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
