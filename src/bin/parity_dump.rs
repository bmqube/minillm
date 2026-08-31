//! Dump MiniLLM final-position logits for a fixed prompt set, so they can be
//! compared against a reference implementation. Pairs with `benchmarks/parity.py`.
//!
//! ```text
//! cargo run --release --bin parity_dump -- [PROMPTS_FILE] [OUT_JSON] [MODEL]
//! ```
//!
//! Defaults: `benchmarks/prompts.txt benchmarks/minillm_logits.json benchmarks/gpt2`.
//! `MODEL` is a local directory (with `config.json`, `tokenizer.json`,
//! `model.safetensors`) or a Hub id like `openai-community/gpt2`. Using a local
//! directory keeps MiniLLM and the Python reference on byte-identical weights.
//!
//! Output JSON: `{ "model": <ref>, "entries": [ { "prompt", "input_ids", "logits" } ] }`
//! where `logits` is the raw (pre-softmax) vector for the position after the
//! last prompt token, and `<ref>` is what `parity.py` should load as the
//! reference (an absolute path for a local dir, otherwise the Hub id).

use std::io::Write;
use std::path::Path;

use candle_core::{IndexOp, Tensor};
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let prompts_path = args
        .next()
        .unwrap_or_else(|| "benchmarks/prompts.txt".to_string());
    let out_path = args
        .next()
        .unwrap_or_else(|| "benchmarks/minillm_logits.json".to_string());
    let model = args.next().unwrap_or_else(|| "benchmarks/gpt2".to_string());

    // What the Python side should feed to `from_pretrained`: an absolute path if
    // this is a local directory, otherwise the Hub id verbatim.
    let model_ref = match std::fs::canonicalize(&model) {
        Ok(p) if p.is_dir() => {
            let s = p.to_string_lossy().replace('\\', "/");
            // strip Windows extended-length prefix (\\?\ -> //?/)
            s.strip_prefix("//?/").unwrap_or(&s).to_string()
        }
        _ => model.clone(),
    };

    let dev = device::best();
    eprintln!("device: {dev:?}  model: {model}  ref: {model_ref}");
    let (net, tokenizer) = loader::load(&model, &dev)?;

    let raw = std::fs::read_to_string(&prompts_path)?;
    let prompts: Vec<&str> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"model\": {model_ref:?},\n"));
    json.push_str("  \"entries\": [\n");

    for (i, prompt) in prompts.iter().enumerate() {
        let ids = tokenizer.encode(*prompt, true)?.get_ids().to_vec();
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        let logits: Vec<f32> = net
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

    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::File::create(&out_path)?.write_all(json.as_bytes())?;
    eprintln!("wrote {} prompts -> {out_path}", prompts.len());
    Ok(())
}
