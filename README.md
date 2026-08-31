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
- **Weights are fp32.** No weight quantization.
- **KV cache.** `GPT2Model::forward_with_cache` decodes one token per step
  against a per-layer key/value cache — O(n) instead of the O(n²) full-recompute
  `forward`. ~14× faster greedy decode on CPU at seq 64→192, and the cached path
  still matches HuggingFace to fp32 noise (see [benchmarks](benchmarks/)). The
  plain `forward` is kept as the reference and pre-cache baseline.
- **int8 KV cache.** `KvQuant::Int8` stores cached K/V as per-token symmetric
  int8 — 3.76× smaller retained cache for effectively no perplexity change
  (Δ ≈ +0.008 on WikiText-2), ~10–30% slower decode. int4 is not done yet.
- Larger GPT-2 sizes (`gpt2-medium/large/xl`) share the architecture and should
  load, but only the 124M base model is regularly exercised.
- Inference only — no training.

## Architecture

```
src/
├── lib.rs          library root
├── main.rs         CLI demo (loads gpt2, generates 50 tokens)
├── loader.rs       download config + tokenizer + safetensors from the HF Hub
├── config.rs       GPT2Config + analytic parameter / KV-cache-byte counts
├── model.rs        GPT2Model: forward + forward_with_cache, embeddings, blocks, offset causal mask
├── transformers.rs TransformerBlock: pre-LN attention + MLP with residuals, cache-aware variant
├── attention.rs    MultiHeadAttention: fused QKV, scaled dot-product, full + incremental paths
├── activations.rs  tanh-approx GELU (matches GPT-2's gelu_new)
├── generation.rs   SamplingConfig + sample(): greedy / temperature / top-k / top-p
├── kv_cache.rs     KvCache / LayerKvCache: per-layer K/V cache, fp32 or per-token int8
└── device.rs       pick CUDA if built with --features cuda, else CPU

src/bin/
├── bench.rs        throughput + size benchmark, no-cache vs KV-cache decode
├── sweep.rs        seq-len throughput sweep (cache off/on, fp32/int8) → CSV
├── ppl.rs          sliding-window perplexity on a text file (--kv-quant off|int8)
├── memprobe.rs     peak-RSS probe for one decode path (KV-cache memory cost)
└── parity_dump.rs  dump logits for the parity check (--cache exercises the cache path)

examples/
└── generate.rs     minimal library-usage example (KV-cache decode loop)

tests/
└── cache_parity.rs KV-cache output == full-recompute output (ignored; needs local weights)

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

On `x86_64-pc-windows-gnu`, `aws-lc-sys` (pulled in via `hf-hub`) compiles
against the MSYS2 winpthreads headers but rustup's bundled `self-contained`
mingw libs are older and lack `nanosleep64`, so the link fails. `.cargo/config.toml`
sets `-C link-self-contained=no` for that target, which hands CRT + winpthreads
resolution to the MSYS2 `gcc` driver (whose `libwinpthread` has the symbol).
Keep MSYS2 `mingw-w64-x86_64-toolchain` on `PATH`. See the comments in that file
for the MSVC `link.exe`-shadowing case.

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
use minillm::kv_cache::KvCache;
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dev = device::best();
    let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;

    let mut ids = tokenizer.encode("The future of AI is", true)?.get_ids().to_vec();
    let cfg = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.95) };

    // Prefill the prompt, then decode one token per step against the cache.
    let mut cache = KvCache::new(model.config().n_layer);
    let prompt = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
    let mut logits = model.forward_with_cache(&prompt, &mut cache)?;

    for _ in 0..40 {
        let next = sample(&logits, &cfg)?;
        print!("{}", tokenizer.decode(&[next], false)?);
        ids.push(next);
        let step = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        logits = model.forward_with_cache(&step, &mut cache)?;
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
cargo run --release --bin bench -- benchmarks/gpt2 64 128        # no-cache vs KV-cache decode
cargo run --release --bin parity_dump -- --cache                 # dump cache-path logits
python benchmarks/parity.py benchmarks/minillm_logits.json       # vs transformers
```

Measured on a Ryzen 5 5600G, CPU, fp32, `openai-community/gpt2` (124M):

| | |
|---|---|
| prefill (64 tok) | ~360–420 tok/s |
| decode (128 steps, seq 64→192), no cache | ~3 tok/s |
| decode (128 steps, seq 64→192), **KV cache** | **~43 tok/s (≈14×)** |
| peak RSS | ~977 MiB |
| parity vs HF Transformers (24 prompts), forward | mean cos `1.00000`, top-1 `24/24`, mean MSE `2.0e-9` |
| parity vs HF Transformers (24 prompts), KV-cache path | mean cos `1.00000`, top-1 `24/24`, mean MSE `1.9e-9` |

Numbers come from real runs, not estimates.

## Tests

```bash
cargo test
```

Covers the samplers (greedy = argmax, `top_k = 1` is deterministic, nucleus
keeps the dominant token, ids stay in range), the analytic parameter /
KV-cache-byte counts, and the `KvCache` container (append grows the sequence
axis, `reset`, length tracking, int8 round-trip error bound). These do not
require downloading a model.

The end-to-end KV-cache check needs GPT-2 weights in `benchmarks/gpt2/` and is
`#[ignore]`d by default:

```bash
cargo test --test cache_parity -- --ignored
```

It asserts that greedy decoding with the cache produces the exact same token ids
as the full-recompute `forward`.

## Roadmap

- ~~KV cache~~ — done (`forward_with_cache`, ~14× faster CPU decode, parity-checked)
- ~~int8 KV-cache quantization~~ — done (per-token int8, perplexity + latency + memory sweep)
- int4 KV-cache quantization (packed, per-head scale + zero-point)
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
