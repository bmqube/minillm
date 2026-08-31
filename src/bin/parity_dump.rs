//! Dump MiniLLM final-position logits for a fixed prompt set, so they can be
//! compared against a reference implementation. Pairs with `benchmarks/parity.py`.
//!
//! ```text
//! cargo run --release --bin parity_dump -- [PROMPTS_FILE] [OUT_JSON] [MODEL_ID]
//! ```
//!
//! Defaults: `benchmarks/prompts.txt benchmarks/minillm_logits.json openai-community/gpt2`.
//!
//! Output JSON: `{ "model": <id>, "entries": [ { "prompt", "input_ids", "logits" } ] }`
//! where `logits` is the raw (pre-softmax) vector for the position after the
//! last prompt token.

use candle_core::{IndexOp, Tensor};
use minillm::{device, loader};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let prompts_path = args
        .next()
        .unwrap_or_else(|| "benchmarks/prompts.txt".to_string());
    let out_path = args
        .next()
        .unwrap_or_else(|| "benchmarks/minillm_logits.json".to_string());
    let model_id = args
        .next()
        .unwrap_or_else(|| "openai-community/gpt2".to_string());

    let dev = device::best();
    eprintln!("device: {dev:?}  model: {model_id}");
    let (model, tokenizer) = loader::load(&model_id, &dev)?;

    let raw = std::fs::read_to_string(&prompts_path)?;
    let prompts: Vec<&str> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"model\": {model_id:?},\n"));
    json.push_str("  \"entries\": [\n");

    for (i, prompt) in prompts.iter().enumerate() {
        let ids = tokenizer.encode(*prompt, true)?.get_ids().to_vec();
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        let logits: Vec<f32> = model
            .forward(&input)?
            .i((0, ids.len() - 1))?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;

        json.push_str("    {");
        json.push_str(&format!("\"prompt\": {prompt:?}, "));
        json.push_str(&format!("\"input_ids\": {ids:?}, "));
        json.push_str(&format!("\"logits\": {logits:?}"));
        json.push('}');
        if i + 1 < prompts.len() {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n}\n");

    std::fs::File::create(&out_path)?.write_all(json.as_bytes())?;
    eprintln!("wrote {} prompts -> {out_path}", prompts.len());
    Ok(())
}
