//! Basic text generation example using MiniLLM

use minillm::inference::InferenceEngine;

fn main() -> minillm::Result<()> {
    println!("🚀 MiniLLM Text Generation Example");
    
    // Create inference engine
    println!("📥 Loading GPT-2 model...");
    let engine = InferenceEngine::new("openai-community/gpt2")?;
    
    // Display model info
    println!("✅ Model loaded successfully!");
    engine.config().print_summary();
    
    // Generate text
    let prompt = "The future of artificial intelligence is";
    println!("\n🎯 Prompt: \"{}\"", prompt);
    
    println!("🔄 Generating text...");
    let generated = engine.generate(prompt, 20)?;
    
    println!("✨ Generated text:\n{}", generated);
    
    Ok(())
}
