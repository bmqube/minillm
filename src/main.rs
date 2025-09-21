//! MiniLLM - A Mini Transformer Inference Engine
//! 
//! Simple example showing basic usage of the inference engine.
//! For more examples, see the examples/ directory.

use minillm::inference::InferenceEngine;

fn main() -> minillm::Result<()> {
    println!("🚀 MiniLLM - Mini Transformer Inference Engine");
    
    // Load the model
    println!("📥 Loading GPT-2 model...");
    let engine = InferenceEngine::new("openai-community/gpt2")?;
    println!("✅ Model loaded successfully!");
    
    // Example generation
    let prompt = "The future of artificial intelligence";
    println!("\n🎯 Generating text for: \"{}\"", prompt);
    
    let generated = engine.generate(prompt, 10)?;
    println!("✨ Result: {}", generated);
    
    println!("\n💡 Try running the examples for more features:");
    println!("   cargo run --example basic_generation");
    println!("   cargo run --example interactive_chat");
    println!("   cargo run --example tokenization");
    
    Ok(())
}
