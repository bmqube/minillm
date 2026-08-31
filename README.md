# MiniLLM

A small GPT-2 inference engine written in Rust. It loads GPT-2 checkpoints from
the HuggingFace Hub and runs autoregressive text generation on CPU or CUDA.

## Two implementations

| Branch | What it is |
|---|---|
| **`main`** (this branch) | Built on [`candle`](https://github.com/huggingface/candle) for tensor ops and the nn primitives. GPU support, less low-level code to maintain. This is the base for ongoing work. |
| **[`scratch`](https://github.com/bmqube/minillm/tree/scratch)** | The original **from-scratch** version: a hand-written tensor library (`src/tensor.rs`, ~1D–4D, matmul / softmax / layernorm) with no ML dependencies, plus the full GPT-2 forward pass on top of it. CPU only. Kept as a reference for the ground-up implementation. |

If you want to see the transformer built without a tensor framework, read the
`scratch` branch. If you want to run models, use `main`.

## Status and scope

- **Works:** GPT-2 (`openai-community/gpt2`) weight loading, forward pass,
  greedy / temperature / top-k / top-p sampling, CPU and CUDA execution.
- **fp32 only.** No quantization.
- **No KV cache yet.** Each generation step recomputes the whole sequence, so
  decoding is O(n²) in sequence length. This is the next planned change; the
  [benchmarks](benchmarks/) establish the pre-cache baseline.
- Larger GPT-2 sizes (`gpt2-medium/large/xl`) share the architecture and should
  load, but only the 124M base model is regularly exercised.
- Inference only — no training.

## Architecture

```
src/
├── lib.rs          library root
├── main.rs         CLI demo (loads gpt2, generates 50 tokens)
├── loader.rs       download config + tokenizer + safetensors from the HF Hub
├── config.rs       GPT2Config + analytic parameter count
├── model.rs        GPT2Model: embeddings, blocks, final norm, LM head, causal mask
├── transformers.rs TransformerBlock: pre-LN attention + MLP with residuals
├── attention.rs    MultiHeadAttention: fused QKV, scaled dot-product, causal mask
├── activations.rs  tanh-approx GELU (matches GPT-2's gelu_new)
├── generation.rs   SamplingConfig + sample(): greedy / temperature / top-k / top-p
└── device.rs       pick CUDA if built with --features cuda, else CPU

src/bin/
├── bench.rs        throughput + size benchmark
└── parity_dump.rs  dump logits for the parity check

examples/
└── generate.rs     minimal library-usage example

benchmarks/         prompt set, parity script, methodology + results
```

## Build

CPU build (default, no CUDA toolkit needed):

```bash
cargo build --release
```

GPU build (requires a CUDA toolkit and a compatible driver):

```bash
cargo build --release --features cuda
```

Rust 1.74+ recommended. Some dependencies compile native code (`ring`,
`aws-lc-rs`, `onig_sys`), so a C toolchain is required: on Linux/macOS the
system compiler is enough; on Windows use the MSVC build tools, or MSYS2
`mingw-w64-gcc` with the `x86_64-pc-windows-gnu` Rust target.

## Usage

### CLI

```bash
cargo run --release                       # CPU
cargo run --release --features cuda        # GPU
```

Downloads `openai-community/gpt2` on first run and prints a 50-token completion
of a fixed prompt, followed by a tok/s line on stderr.

### Library

```rust
use candle_core::Tensor;
use minillm::generation::{sample, SamplingConfig};
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dev = device::best();
    let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;

    let mut ids = tokenizer.encode("The future of AI is", true)?.get_ids().to_vec();
    let cfg = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.95) };

    for _ in 0..40 {
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        let next = sample(&model.forward(&input)?, &cfg)?;
        print!("{}", tokenizer.decode(&[next], false)?);
        ids.push(next);
    }
    Ok(())
}
```

Run the same thing as an example:

```bash
cargo run --release --example generate -- "Once upon a time"
```

### Sampling options

`SamplingConfig` controls `sample(&logits, &cfg)`:

| field | effect |
|---|---|
| `temperature: f64` | divides logits before softmax; `<= 1e-6` selects greedy (argmax) |
| `top_k: Option<usize>` | keep only the `k` highest-scoring tokens |
| `top_p: Option<f64>` | nucleus: keep the shortest ranked prefix whose probability mass reaches `p` |

Sampling runs on the CPU over a `Vec<f32>` — one token per step is cheap and the
scalar code is easy to verify.

## Model loading

Weights, `config.json` and `tokenizer.json` are fetched via `hf-hub` and cached
under `~/.cache/huggingface`. For gated or private repos, provide a token:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env      # loaded via dotenvy
# or: export HF_TOKEN=hf_your_token_here
```

## Benchmarks

See [`benchmarks/`](benchmarks/) for the harness, methodology and full results.

```bash
# one-time: fetch GPT-2 weights into a local dir (see benchmarks/README.md)
cargo run --release --bin bench -- benchmarks/gpt2 64 128     # tok/s + size
cargo run --release --bin parity_dump                         # dump logits
python benchmarks/parity.py benchmarks/minillm_logits.json    # vs transformers
```

Measured on a Ryzen 5 5600G, CPU, fp32, `openai-community/gpt2` (124M):

| | |
|---|---|
| prefill (64 tok) | ~366 tok/s |
| decode (128 steps, seq 64→192) | ~3.3 tok/s — **no KV cache yet** |
| peak RSS | ~977 MiB |
| parity vs HF Transformers (24 prompts) | mean cos `1.00000`, top-1 `24/24`, mean MSE `2.0e-9` |

Numbers come from real runs, not estimates.

## Tests

```bash
cargo test
```

Covers the samplers (greedy = argmax, `top_k = 1` is deterministic, nucleus
keeps the dominant token, ids stay in range) and the analytic parameter count.
Tests do not require downloading a model.

## Roadmap

- KV cache (single biggest inference speedup; benchmarked against the current baseline)
- INT8 / INT4 weight-only quantization with a perplexity + latency + memory sweep
- Verify and benchmark `gpt2-medium/large/xl`
- Batched generation

## License

MIT — see [LICENSE](LICENSE).

## Author

BM Monjur Morshed — [@bmqube](https://github.com/bmqube)

## Acknowledgments

- [candle](https://github.com/huggingface/candle) for tensors and nn primitives
- HuggingFace for model weights and the tokenizers library
- Inspired by Andrej Karpathy's educational GPT implementations
