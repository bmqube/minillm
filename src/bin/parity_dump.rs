//! Dump MiniLLM final-position logits for a fixed prompt set, so they can be
//! compared against a reference implementation. Pairs with `benchmarks/parity.py`.
//!
//! ```text
//! cargo run --release --bin parity_dump -- [--cache] [PROMPTS_FILE] [OUT_JSON] [MODEL]
//! ```
//!
//! Defaults: `benchmarks/prompts.txt benchmarks/minillm_logits.json benchmarks/gpt2`.
//! `MODEL` is a local directory (with `config.json`, `tokenizer.json`,
//! `model.safetensors`) or a Hub id like `openai-community/gpt2`. Using a local
//! directory keeps MiniLLM and the Python reference on byte-identical weights.
//!
//! With `--cache`, each prompt is run through the KV-cache path instead of the
//! plain forward: all but the last token are prefilled, then the last token is
//! fed as a single decode step and its logits are dumped. This checks that the
//! offset causal mask and the single-token decode path still match the
//! reference. Without it, the plain full-sequence `forward` is used.
//!
//! Output JSON: `{ "model": <ref>, "entries": [ { "prompt", "input_ids", "logits" } ] }`
//! where `logits` is the raw (pre-softmax) vector for the position after the
//! last prompt token, and `<ref>` is what `parity.py` should load as the
//! reference (an absolute path for a local dir, otherwise the Hub id).

use std::io::Write;
use std::path::Path;

use candle_core::{IndexOp, Tensor};
use minillm::generation::Generator;
use minillm::models::CausalLM;
use minillm::{device, loader};
use serde::Serialize;

#[derive(Serialize)]
struct Entry {
    prompt: String,
    input_ids: Vec<u32>,
    logits: Vec<f32>,
}

#[derive(Serialize)]
struct Dump {
    model: String,
    entries: Vec<Entry>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let mut positional: Vec<String> = Vec::new();
    let mut use_cache = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--cache" => use_cache = true,
            _ => positional.push(arg),
        }
    }
    let mut positional = positional.into_iter();
    let prompts_path = positional
        .next()
        .unwrap_or_else(|| "benchmarks/prompts.txt".to_string());
    let out_path = positional
        .next()
        .unwrap_or_else(|| "benchmarks/minillm_logits.json".to_string());
    let model = positional
        .next()
        .unwrap_or_else(|| "benchmarks/gpt2".to_string());

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
    eprintln!(
        "device: {dev:?}  model: {model}  ref: {model_ref}  path: {}",
        if use_cache { "KV cache" } else { "forward" }
    );
    let (net, tokenizer) = loader::load(&model, &dev)?;
    let net = net.as_ref();
    eprintln!(
        "arch: {}  precision: {}",
        net.meta().architecture,
        net.precision()
    );

    let raw = std::fs::read_to_string(&prompts_path)?;
    let prompts: Vec<&str> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut entries = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let ids = tokenizer.encode(*prompt, true)?.get_ids().to_vec();
        let logits: Vec<f32> = if use_cache {
            last_logits_cached(net, &ids)?
        } else {
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
            net.forward_last(&input)?
                .i(0)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?
        };
        entries.push(Entry {
            prompt: prompt.to_string(),
            input_ids: ids,
            logits,
        });
    }

    let dump = Dump {
        model: model_ref,
        entries,
    };

    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string_pretty(&dump)?;
    std::fs::File::create(&out_path)?.write_all(json.as_bytes())?;
    eprintln!("wrote {} prompts -> {out_path}", prompts.len());
    Ok(())
}

/// Prefill `ids[..n-1]` through the cache, then feed `ids[n-1]` as one decode
/// step and return that step's logits. For a single-token prompt this is just a
/// one-token prefill.
fn last_logits_cached(
    net: &dyn CausalLM,
    ids: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let mut generator = Generator::new(net);
    let split = ids.len().saturating_sub(1);

    let logits = if split == 0 {
        generator.prefill(ids)?.clone()
    } else {
        generator.prefill(&ids[..split])?;
        generator.feed(ids[split])?.clone()
    };

    Ok(logits.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
}
