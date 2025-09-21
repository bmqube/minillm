//! Tokenization example showing how to work with tokens

use minillm::inference::InferenceEngine;

fn main() -> minillm::Result<()> {
    println!("🔤 MiniLLM Tokenization Example");
    
    // Load model
    let engine = InferenceEngine::new("openai-community/gpt2")?;
    
    let texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "Transformer models revolutionized natural language processing.",
    ];
    
    for text in texts {
        println!("\n📝 Text: \"{}\"", text);
        
        // Tokenize
        let tokens = engine.tokenize(text)?;
        println!("🔢 Token count: {}", tokens.len());
        println!("🎯 Token IDs: {:?}", tokens);
        
        // Decode back
        let decoded = engine.decode(&tokens)?;
        println!("🔄 Decoded: \"{}\"", decoded);
        
        // Verify round-trip
        if text.trim() == decoded.trim() {
            println!("✅ Round-trip successful");
        } else {
            println!("⚠️  Round-trip mismatch");
        }
    }
    
    Ok(())
}
