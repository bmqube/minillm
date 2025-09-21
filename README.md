# MiniLLM 🤖

A lightweight, efficient transformer inference engine written in Rust. MiniLLM provides a clean, well-documented implementation of GPT-2 style transformer models with support for text generation.

## ✨ Features

- **🚀 Fast Inference**: Efficient tensor operations using ndarray
- **🔒 Memory Safe**: Written in Rust with zero-copy operations where possible  
- **📦 Easy to Use**: High-level API for quick integration
- **🎯 Well Tested**: Comprehensive examples and documentation
- **🔧 Extensible**: Modular architecture for easy customization
- **🤖 GPT-2 Compatible**: Load and run GPT-2 models from HuggingFace
- **📊 SafeTensors Support**: Fast and secure model weight loading

## 🏗️ Architecture

```
src/
├── lib.rs          # Library entry point and public API
├── main.rs         # Simple CLI example (clean 27 lines)
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

### Command Line

```bash
# Run the main example
cargo run

# Run specific examples  
cargo run --example basic_generation
cargo run --example interactive_chat
cargo run --example tokenization
```

## 📋 Requirements

- Rust 1.70+
- HuggingFace token (optional, for private models)

Set your HuggingFace token:
```bash
echo "HF_TOKEN=your_token_here" > .env
```

## 🔧 Dependencies

- `ndarray` - Tensor operations
- `safetensors` - Model weight loading
- `tokenizers` - Text tokenization
- `hf-hub` - HuggingFace model downloading
- `serde` - Configuration parsing

## 📖 API Documentation

### InferenceEngine

The main high-level interface:

```rust
// Create engine
let engine = InferenceEngine::new("openai-community/gpt2")?;

// Generate text
let result = engine.generate("prompt", max_tokens)?;

// Tokenization
let tokens = engine.tokenize("text")?;
let text = engine.decode(&tokens)?;

// Get model info
let config = engine.config();
```

### Low-Level Components

For custom implementations, you can use the individual components:

- `GPTModel` - Complete transformer model
- `TransformerBlock` - Individual transformer layers  
- `MultiHeadAttention` - Attention mechanism
- `MLP` - Feed-forward networks
- `Tensor` - Mathematical operations

## 🎯 Examples

### Basic Generation
```bash
cargo run --example basic_generation
```
Demonstrates simple text generation with model configuration display.

### Interactive Chat
```bash
cargo run --example interactive_chat
```
Interactive command-line chat interface with the model.

### Tokenization
```bash
cargo run --example tokenization
```
Shows tokenization, encoding/decoding, and round-trip verification.

## 📊 Performance

MiniLLM is designed for inference efficiency:

- **Memory**: ~1GB RAM for GPT-2 (117M parameters)
- **Speed**: ~10-50 tokens/second (CPU, varies by hardware)
- **Accuracy**: Identical outputs to reference implementations
- **Models**: Currently supports GPT-2 architecture

## 🛠️ Development

```bash
# Clone and build
git clone https://github.com/bmqube/minillm
cd minillm
cargo build --release

# Run tests
cargo test

# Check examples
cargo check --examples

# Generate documentation
cargo doc --open
```

## 📚 Architecture Details

### Transformer Implementation
- **Multi-head attention** with causal masking
- **Feed-forward networks** with GELU activation
- **Layer normalization** and residual connections
- **Position and token embeddings**

### Tensor Operations
- Dynamic 1D-4D tensor support
- Optimized matrix multiplication
- Element-wise operations (add, softmax, layer_norm)
- Memory-efficient implementations

### Model Loading
- SafeTensors format support
- Automatic model downloading from HuggingFace
- Configuration parsing and validation
- Error handling with detailed messages

## ✅ Current Status

- ✅ **Core Architecture**: Complete GPT-2 implementation
- ✅ **Inference Engine**: High-level API ready
- ✅ **Examples**: Comprehensive usage examples
- ✅ **Documentation**: Well-documented codebase
- ✅ **Testing**: All components tested and working

## 🗺️ Roadmap

- [ ] **Performance**: GPU acceleration support
- [ ] **Models**: Support for larger GPT variants
- [ ] **Features**: Beam search and sampling options
- [ ] **Optimization**: Quantization and pruning
- [ ] **Integration**: Python bindings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's educational implementations
- Built on the excellent Rust ecosystem (ndarray, tokenizers, etc.)
- Model weights from HuggingFace transformers library

## 👨‍💻 Author

**BM Monjur Morshed**  
- GitHub: [@bmqube](https://github.com/bmqube)
- Project: [minillm](https://github.com/bmqube/minillm)
