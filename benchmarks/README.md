# Benchmarks

Reproducible throughput, memory and correctness checks for MiniLLM.

Everything here is deterministic given a set of weights and the fixed prompt set
in [`prompts.txt`](prompts.txt). Re-run after any change to the model or
generation code and update the tables below.

## 0. Get the weights (once)

The benchmarks load GPT-2 from a local directory so MiniLLM and the Python
reference use byte-identical weights (and so nothing depends on a shared HF
cache). `benchmarks/gpt2/` is git-ignored; populate it once:

```bash
mkdir -p benchmarks/gpt2 && cd benchmarks/gpt2
base=https://huggingface.co/openai-community/gpt2/resolve/main
for f in config.json tokenizer.json model.safetensors; do curl -sL -o "$f" "$base/$f"; done
```

Both binaries also accept a Hub id (`openai-community/gpt2`) instead of the
directory, if you'd rather use the HF cache.

## 1. Throughput and size

```bash
cargo run --release --bin bench -- benchmarks/gpt2 64 128            # CPU
cargo run --release --features cuda --bin bench -- benchmarks/gpt2 64 128   # GPU
```

Arguments: `MODEL PREFILL_TOKENS DECODE_STEPS`.

Reports: load time, analytic parameter count, fp32 weight memory, prefill
latency (tok/s), and greedy decode throughput (tok/s). It does **not** measure
peak RAM itself — wrap the command:

| OS | command | field |
|---|---|---|
| Linux | `/usr/bin/time -v <cmd>` | `Maximum resident set size` |
| macOS | `/usr/bin/time -l <cmd>` | `maximum resident set size` |
| Windows | `Get-Process bench \| Select-Object PeakWorkingSet64` | bytes |

> There is no KV cache yet: every decode step recomputes the whole sequence, so
> decode tok/s falls as the sequence grows. These numbers are the **baseline**
> for the planned KV-cache work — keep the same `PREFILL`/`DECODE` args when
> comparing before/after.

### Results — `openai-community/gpt2` (124M), fp32

Environment: AMD Ryzen 5 5600G (6C/12T), 28 GiB RAM, Windows 11 Pro ·
rustc 1.98.0 `x86_64-pc-windows-gnu` (MSYS2 mingw-w64 gcc 16.2.0) ·
candle 0.11.0 · `--release` · weights loaded from `benchmarks/gpt2/`.

| metric | CPU | GPU |
|---|---|---|
| hardware | Ryzen 5 5600G, 28 GiB | not measured (no CUDA toolkit on this box) |
| load time (local files) | 0.40 s | — |
| parameters | 124,439,808 | — |
| fp32 weights | ~0.46 GiB | — |
| prefill, 64 tok | 175 ms → **366 tok/s** (360–420 across runs) | — |
| decode, 128 steps (seq 64→192) | **3.3 tok/s** (~38 s) — no KV cache | — |
| peak RSS | **977 MiB** | — |

_GPU row: run the `--features cuda` command on a machine with the CUDA toolkit._

## 2. Output parity vs HuggingFace Transformers

Confirms MiniLLM computes the same thing as the reference implementation, up to
fp32 rounding.

```bash
python -m pip install numpy torch transformers        # once

cargo run --release --bin parity_dump                 # -> benchmarks/minillm_logits.json
python benchmarks/parity.py benchmarks/minillm_logits.json
```

`parity_dump` records the exact `input_ids` it used and the absolute path to the
weights directory; `parity.py` loads the HF reference from that same directory
with `local_files_only=True` and feeds it the same ids — so the check is
tokenizer- and cache-independent. It compares the raw logits for the position
right after the last prompt token.

Metrics: `mse`, `mae`, `max|Δ|`, cosine similarity,
`KL(softmax(hf) || softmax(mini))`, top-1 match, top-5 set overlap.

**Pass criteria** (fp32): `mean cos > 0.9999`, `top1 = n/n`, `mean mse ≲ 1e-3`.
A large KL or any top-1 miss means a real discrepancy (weight transpose,
activation, LayerNorm eps, position ids, tying).

### Results — `openai-community/gpt2`

Reference: transformers 5.16.1, torch 2.13.0+cpu, numpy 2.5.2 · same fp32
`benchmarks/gpt2/` weights.

| aggregate | value |
|---|---|
| prompts | 24 |
| mean mse | **1.996e-09** |
| mean cosine similarity | **1.00000** |
| mean KL(hf‖mini) | **3.984e-10** |
| per-prompt max\|Δ\| (logit) | ~1–2 × 10⁻⁴ |
| top-1 agreement | **24 / 24** |
| top-5 overlap | **120 / 120** |

MiniLLM's GPT-2 forward pass matches HuggingFace to floating-point noise.

## 3. Larger GPT-2 variants (optional)

`bench` and `parity_dump` accept `openai-community/gpt2-medium`, `-large`,
`-xl` (or a local dir for each). Record a row per size once weight loading for
those is verified.

| model | params | load (s) | prefill tok/s | decode tok/s | peak RSS | mean cos vs HF |
|---|---|---|---|---|---|---|
| gpt2 | 124M | 0.40 | 366 | 3.3 | 977 MiB | 1.00000 |
| gpt2-medium | 355M | | | | | |
| gpt2-large | 774M | | | | | |
