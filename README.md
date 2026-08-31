# MiniLLM

[![CI](https://github.com/bmqube/minillm/actions/workflows/ci.yml/badge.svg)](https://github.com/bmqube/minillm/actions/workflows/ci.yml)

A small transformer inference engine written in Rust. It loads decoder-only
checkpoints from the HuggingFace Hub and runs autoregressive text generation on
CPU or CUDA, with a KV cache that can store keys and values at full precision,
per-token int8, or per-token packed int4.

Two architectures, one interface:

| | GPT-2 | Qwen3 |
|---|---|---|
| checkpoints | `openai-community/gpt2{,-medium,-large,-xl}` | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B}` |
| positions | learned `wpe` table | RoPE (`rope_theta` 1e6) |
| normalization | LayerNorm | RMSNorm |
| attention | multi-head, fused QKV | grouped-query, separate Q/K/V |
| Q/K scaling | — | per-head RMSNorm (QK-norm) before RoPE |
| MLP | 4×, tanh-GELU | SwiGLU |
| weight layout | Conv1D `[in, out]` | `nn.Linear` `[out, in]` |
| context | 1024 | 40960 |
| published dtype | fp32 | bf16 |

Both are verified against HuggingFace Transformers to floating-point noise, on
the forward path *and* the KV-cache path (see [Benchmarks](#benchmarks)).

## Two implementations

| Branch | What it is |
|---|---|
| **`main`** (this branch) | Built on [`candle`](https://github.com/huggingface/candle) for tensor ops and the nn primitives. GPU support, less low-level code to maintain. This is the base for ongoing work. |
| **[`scratch`](https://github.com/bmqube/minillm/tree/scratch)** | The original **from-scratch** version: a hand-written tensor library (`src/tensor.rs`, ~1D–4D, matmul / softmax / layernorm) with no ML dependencies, plus the full GPT-2 forward pass on top of it. CPU only. Kept as a reference for the ground-up implementation. |

If you want to see the transformer built without a tensor framework, read the
`scratch` branch. If you want to run models, use `main`.

## Status and scope

- **Works:** GPT-2 and Qwen3 weight loading (single-file *and* sharded
  safetensors), forward pass, greedy / temperature / top-k / top-p sampling,
  CPU and CUDA execution.
- **No weight quantization.** Weights load at the checkpoint's own
  `torch_dtype` — fp32 for GPT-2, bf16 for Qwen3 — or at an explicit
  `Precision`. candle has no CPU bf16 gemm, so a bf16 checkpoint on CPU falls
  back to fp32 with a note; on CUDA it stays bf16.
- **KV cache.** `forward_with_cache` decodes one token per step against a
  per-layer key/value cache — O(n) instead of the O(n²) full-recompute
  `forward`. ~11× faster greedy decode on CPU at seq 64→192 (up to ~31× by seq
  512→544, see the sweep in [benchmarks](benchmarks/)), and the cached path
  still matches HuggingFace to fp32 noise. The plain `forward` is kept as the
  reference and pre-cache baseline; both it and the cached path skip the LM head
  on positions the caller won't use (`forward_last` / `forward_with_cache_last`).
- **Grouped-query attention is respected in the cache.** Only the narrow
  `n_kv_head` tensors are stored; `repeat_kv` expands to the query head count
  per step and is never cached. On Qwen3-0.6B (16 query heads over 8 KV heads)
  that is a 2× smaller cache before any quantization.
- **Quantized KV cache.** `KvQuant::Int8` (per-token symmetric) is 3.76× smaller
  and `KvQuant::Int4` (per-token asymmetric, packed) 6.4× smaller. On GPT-2 int8
  costs effectively nothing (Δppl ≈ +0.008 on WikiText-2) — **but that result
  does not transfer to Qwen3**, which is ~7× more sensitive to int8 and ~16×
  more to int4, to the point where int8 changes the top-1 token on prompts GPT-2
  handles unaffected. See [Benchmarks](#benchmarks).
- Larger sizes (`gpt2-medium/large/xl`, `Qwen3-1.7B` and up) share their
  family's architecture and load through the same path, but only `gpt2` 124M and
  `Qwen3-0.6B` are regularly exercised.
- Inference only — no training.

## Architecture

Architecture-specific code lives under `models/`, behind the `CausalLM` trait.
Everything above it — the decode loop, the cache, the benchmark harnesses — is
generic and never names a concrete model type.

```
src/
├── lib.rs          library root
├── main.rs         CLI demo: [MODEL] [PROMPT], KV-cache decode
├── loader.rs       config parsing, architecture dispatch, sharded safetensors, precision choice
├── dtype.rs        Precision (f32/f16/bf16) + what each device can actually run
├── device.rs       pick CUDA if built with --features cuda, else CPU
├── generation.rs   Generator (decode loop) + SamplingConfig / sample(): greedy / temp / top-k / top-p
├── kv_cache.rs     KvCache / LayerKvCache: per-layer K/V, model precision or per-token int8 / int4
│
├── layers/         primitives shared across architectures
│   ├── activation.rs  tanh-approx GELU (GPT-2's gelu_new) + SiLU (Qwen3's SwiGLU gate)
│   ├── mask.rs        additive causal + sliding-window masks, dtype-aware
│   ├── rope.rs        rotary embeddings, offset-aware for cached decode
│   └── mod.rs         repeat_kv: GQA head expansion
│
└── models/
    ├── mod.rs      CausalLM trait + ModelMeta (layers, q/kv heads, head_dim, ctx, params)
    ├── gpt2/       config, attention (fused QKV, MHA), block (pre-LN + GELU MLP), model
    └── qwen3/      config, attention (GQA + QK-norm + RoPE), block (RMSNorm + SwiGLU), model

src/bin/
├── bench.rs        throughput + size benchmark, no-cache vs KV-cache decode
├── sweep.rs        seq-len sweep (cache off/on, quant, --repeats N, --precision) → CSV
├── ppl.rs          sliding-window perplexity on a text file (--kv-quant off|int8|int4)
├── memprobe.rs     peak-RSS probe for one decode path (KV-cache memory cost)
└── parity_dump.rs  dump logits for the parity check (--cache exercises the cache path)

examples/
└── generate.rs     minimal library-usage example (KV-cache decode loop)

tests/
└── cache_parity.rs cache == full-recompute, incremental == bulk prefill, GQA cache
                    shape, quantization drift — per architecture (ignored; needs weights)

benchmarks/         prompt set, parity scripts, methodology + results, GPU runbook
```

### Adding an architecture

1. `src/models/<name>/` with `config.rs` (deserialize `config.json`, expose a
   `ModelMeta`), `attention.rs`, `block.rs`, `model.rs`.
2. `impl CausalLM for <Name>Model`.
3. One arm in `loader::Architecture` and its `config.json` probe.

Reuse `layers/` for anything that isn't specific to the architecture. Nothing
else in the crate needs to change — the binaries, the cache and the decode loop
are already generic.

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
cargo run --release                                        # gpt2, CPU
cargo run --release --features cuda                         # gpt2, GPU
cargo run --release -- Qwen/Qwen3-0.6B "Once upon a time"   # qwen3
```

Arguments are `[MODEL] [PROMPT]`, where `MODEL` is a Hub id or a local
directory. Downloads on first run and prints a 50-token completion, followed by
a tok/s line on stderr.

### Library

`loader::load` returns a `Box<dyn CausalLM>` — the architecture is picked from
the checkpoint's `config.json`, so the calling code is identical either way.
`Generator` owns the KV cache and drives the decode loop: prefill the prompt
once, then take one token per step.

```rust
use minillm::generation::{Generator, SamplingConfig};
use minillm::{device, loader};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dev = device::best();
    // Swap for "Qwen/Qwen3-0.6B" and nothing below changes.
    let (model, tokenizer) = loader::load("openai-community/gpt2", &dev)?;

    let ids = tokenizer.encode("The future of AI is", true)?.get_ids().to_vec();
    let cfg = SamplingConfig { temperature: 0.8, top_k: Some(40), top_p: Some(0.95) };

    let mut generator = Generator::new(model.as_ref());
    generator.prefill(&ids)?;

    for _ in 0..40 {
        let next = generator.next_token(&cfg)?;
        print!("{}", tokenizer.decode(&[next], false)?);
    }
    Ok(())
}
```

`model.meta()` exposes the shape facts a caller might need — layer count, query
and KV head counts, `head_dim`, context window, parameter count, and
`kv_cache_bytes_per_token_with(quant, precision)` for cache accounting.

### Precision

```rust
use minillm::dtype::Precision;
let (model, tok) = loader::load_with("Qwen/Qwen3-0.6B", &dev, Some(Precision::BF16))?;
```

Omit the override and the checkpoint's own `torch_dtype` is used. candle has no
CPU bf16 gemm, so on CPU a bf16 checkpoint falls back to fp32 with a note, while
an *explicit* `Precision::BF16` on CPU is refused outright rather than failing
later inside a matmul.

Run the same thing as an example:

```bash
cargo run --release --example generate -- "Once upon a time"
```

Other entry points:

| | |
|---|---|
| `Generator::with_quant(model, KvQuant::Int8)` | quantized KV cache |
| `generation::greedy(model, &ids, n, quant)` | one-shot greedy decode |
| `generation::greedy_no_cache(model, &ids, n)` | the O(n²) reference path |
| `generation::sample_with(&logits, &cfg, &mut rng)` | seeded, reproducible sampling |
| `Generator::next_token_with(&cfg, &mut rng)` | same, inside the decode loop |
| `CausalLM::forward_last` / `forward_with_cache_last` | only the final position's logits, skipping the LM head on the rest — what `Generator` and `greedy_no_cache` use internally |
| `loader::load_with(spec, &dev, Some(Precision::BF16))` | pin the weight dtype |
| `model.meta()` | layers, q/kv heads, `head_dim`, context, params, KV bytes/token |

Greedy decoding (`temperature <= 1e-6`, no `top_k`/`top_p`) consumes no
randomness, so every benchmark in this repo is deterministic. For reproducible
*sampled* output, pass a seeded RNG:

```rust
use rand::SeedableRng;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let next = generator.next_token_with(&cfg, &mut rng)?;
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
| prefill (64 tok) | ~485–510 tok/s |
| decode (128 steps, seq 64→192), no cache | ~4.5 tok/s |
| decode (128 steps, seq 64→192), **KV cache** | **~49–50 tok/s (≈11×)** |
| peak RSS | ~977 MiB |
| parity vs HF Transformers (24 prompts), forward | mean cos `1.00000`, top-1 `24/24`, mean MSE `1.8e-9` |
| parity vs HF Transformers (24 prompts), KV-cache path | mean cos `1.00000`, top-1 `24/24`, mean MSE `2.2e-9` |

Numbers come from real runs, not estimates.

## Tests

```bash
cargo test
```

Covers the samplers (greedy = argmax, `top_k = 1` is deterministic, nucleus
keeps the dominant token, ids stay in range), the analytic parameter /
KV-cache-byte counts, and the `KvCache` container (append grows the sequence
axis, `reset`, length tracking, int8/int4 round-trip error bounds). These do not
require downloading a model.

The end-to-end KV-cache checks need GPT-2 weights in `benchmarks/gpt2/` and are
`#[ignore]`d by default:

```bash
cargo test --test cache_parity -- --ignored
```

`cache_matches_no_cache` asserts that greedy decoding with the cache produces the
exact same token ids as the full-recompute `forward`; `quantized_caches_stay_close_to_fp32`
bounds how far int8/int4 prefill logits drift, and that int8 stays closer than int4.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `cargo fmt --check`,
`cargo clippy --all-targets -D warnings`, the test suite and `cargo doc` on every
push and PR.

## Roadmap

- ~~KV cache~~ — done (`forward_with_cache`, ~11× faster CPU decode, parity-checked)
- ~~int8 / int4 KV-cache quantization~~ — done (per-token, perplexity + latency + memory ablation)
- Low-precision attention matmul (turn the memory saving into a peak-RSS + speed win)
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
