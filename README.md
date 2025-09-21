# MiniLLM 🤖

A lightweight, efficient transformer inference engine written in Rust. MiniLLM provides a clean, well-documented implementation of GPT-2 style transformer models with support for text generation.

## ✨ Features

- **� Fast Inference**: Efficient tensor operations using ndarray
- **🔒 Memory Safe**: Written in Rust with zero-copy operations where possible  
- **📦 Easy to Use**: High-level API for quick integration
- **🎯 Well Tested**: Comprehensive examples and documentation
- **🔧 Extensible**: Modular architecture for easy customization

## 🏗️ Architecture

```
src/
├── lib.rs          # Library entry point and public API
├── main.rs         # Simple CLI example
├── inference.rs    # High-level inference engine
├── gpt.rs          # GPT model implementation
├── transformer.rs  # Transformer block components
├── attention.rs    # Multi-head attention mechanism
├── mlp.rs          # Feed-forward network layers
├── tensor.rs       # Tensor operations and math
├── weights.rs      # Model weight loading (SafeTensors)
└── config.rs       # Model configuration handling

examples/
├── basic_generation.rs  # Simple text generation
├── interactive_chat.rs  # Interactive chat interface
└── tokenization.rs      # Tokenization examples
```

## 🚀 Quick Start

### Library Usage

```rust
use minillm::inference::InferenceEngine;

fn main() -> minillm::Result<()> {
    // Load a GPT-2 model
    let engine = InferenceEngine::new("openai-community/gpt2")?;
    
    // Generate text
    let prompt = "The future of AI is";
    let generated = engine.generate(prompt, 20)?;
    
    println!("Generated: {}", generated);
    Ok(())
}
```

MiniLLM is a lightweight inference engine designed for running language models efficiently. The goal is to provide a simple, fast, and memory-efficient solution for LLM inference.

## Features (Planned)

- [ ] Model loading and inference
- [ ] Support for multiple model formats
- [ ] Memory-efficient execution
- [ ] CPU and GPU acceleration
- [ ] Simple API for integration

## Installation

This crate is not yet published to crates.io. Once available, you can install it with:

```bash
cargo add minillm
```

## Usage

Documentation and usage examples will be provided as the project develops.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

BM Monjur Morshed
