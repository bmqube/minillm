//! Interactive chat example with MiniLLM

use minillm::inference::InferenceEngine;
use std::io::{self, Write};

fn main() -> minillm::Result<()> {
    println!("💬 MiniLLM Interactive Chat");
    println!("Type 'quit' to exit\n");
    
    // Load model
    println!("📥 Loading model...");
    let engine = InferenceEngine::new("openai-community/gpt2")?;
    println!("✅ Ready to chat!\n");
    
    loop {
        print!("You: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("quit") {
            break;
        }
        
        if prompt.is_empty() {
            continue;
        }
        
        print!("AI: ");
        io::stdout().flush()?;
        
        match engine.generate(prompt, 15) {
            Ok(response) => {
                // Extract only the new part (remove the original prompt)
                let new_text = response.trim_start_matches(prompt).trim();
                println!("{}\n", new_text);
            }
            Err(e) => {
                println!("Error: {}\n", e);
            }
        }
    }
    
    println!("👋 Goodbye!");
    Ok(())
}
