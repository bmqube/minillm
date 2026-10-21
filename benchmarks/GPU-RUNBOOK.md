# GPU runbook

Everything in `benchmarks/README.md` was measured on CPU. This file is the exact
sequence to re-run it on a CUDA box and fill in the GPU columns.

Written for an **RTX 5090 (32 GB, Blackwell, compute capability 12.0)**; any
CUDA GPU works, the only thing that changes is which models fit.

---

## 0. Prerequisites

| | |
|---|---|
| NVIDIA driver | 570+ for Blackwell (`nvidia-smi` should print the card) |
| CUDA toolkit | 12.8 or newer for `sm_120`; **12.0–12.6 will not compile for a 5090** |
| Rust | 1.89+ |
| C toolchain | Linux: system gcc. Windows: MSVC build tools, or MSYS2 mingw-w64 (see the root README) |

Check the toolkit is visible:

```bash
nvcc --version          # must report >= 12.8 for sm_120
nvidia-smi              # must list the GPU
```

If `nvcc` reports an older version, candle will build kernels the 5090 cannot
run and every launch fails at runtime with `no kernel image is available for
execution on the device`. That is a toolkit problem, not a code problem.

---

## 1. Build

```bash
cargo build --release --features cuda
```

First build compiles CUDA kernels and takes noticeably longer than the CPU
build. `device::best()` then picks GPU 0 automatically — there is no runtime
flag. A binary built **without** `--features cuda` silently runs on CPU, so if
throughput looks like the CPU numbers, check the build flags first.

Select a specific GPU on a multi-GPU box with `CUDA_VISIBLE_DEVICES=0`.

### 1.1 First-run checklist

The hardware-adaptive paths (dtype probe, preallocated KV cache read through
cuBLAS as a strided view, grouped-query attention without `repeat_kv`, greedy
argmax on the device) were written and tested on CPU only. Before recording any
number, confirm they behave on the GPU:

```bash
./target/release/bench benchmarks/qwen3-0.6b 16 8
```

1. The first line is the execution plan. It must read `device=cuda`,
   `half=bf16+f16` and `argmax=device`.
   - `device=cpu`: the build lacks `--features cuda`.
   - `half=f16` or `half=none` on an Ampere-or-newer card: the bf16 probe in
     `hardware::supports` failed. The model will load at fp32; record the
     card and driver and report it.
2. `precision : bf16` and no `note: checkpoint is bf16 ...` line.
3. Then run the §4.2 gate. A `MatMulNonContiguous` error there means cuBLAS
   rejected the cache view, which is a bug in this crate, not a setup
   problem; record the full message.

---

## 2. Fetch the checkpoints

`benchmarks/*/` is git-ignored; populate it once per machine.

```bash
# GPT-2 124M (fp32, the reference architecture)
mkdir -p benchmarks/gpt2 && cd benchmarks/gpt2
base=https://huggingface.co/openai-community/gpt2/resolve/main
for f in config.json tokenizer.json model.safetensors; do curl -sL -O "$base/$f"; done
cd ../..

# Qwen3-0.6B (bf16, small enough to iterate on)
mkdir -p benchmarks/qwen3-0.6b && cd benchmarks/qwen3-0.6b
base=https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
for f in config.json tokenizer.json model.safetensors; do curl -sL -O "$base/$f"; done
cd ../..
```

Verify the GPT-2 bytes match the hashes in `benchmarks/README.md` §0 — every
CPU number in that file was produced from them, so a GPU/CPU comparison is only
valid on identical weights.

**Larger Qwen3 checkpoints are sharded** (`model-0000N-of-0000M.safetensors` plus
`model.safetensors.index.json`). The loader handles them, but you must download
every shard *and* the index. Easiest:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", local_dir="benchmarks/qwen3-8b",
                  allow_patterns=["*.json", "*.safetensors"])
PY
```

### What fits in 32 GB

Weights only; add the KV cache and activations on top. bf16 unless noted.

| model | params | bf16 weights | fits with long context? |
|---|---|---|---|
| Qwen3-0.6B | 0.6 B | ~1.2 GB | yes, trivially |
| Qwen3-1.7B | 1.7 B | ~3.4 GB | yes |
| Qwen3-4B | 4 B | ~8 GB | yes |
| **Qwen3-8B** | 8 B | ~16 GB | **yes — the headline config** |
| Qwen3-14B | 14 B | ~28 GB | tight; short context only |
| Qwen3.5-0.8B | 0.75 B text | ~1.5 GB | yes, trivially |
| **Qwen3.5-9B** | ~9 B text | ≤ 19 GB | **yes**, with a KV cache in only 8 of its 32 layers |
| Qwen3.5 / 3.6 / 3.8-27B | 27.8 B | ~56 GB | no — needs weight quantization |
| GPT-2 xl | 1.5 B | ~6 GB at fp32 | yes |

Qwen3.5 checkpoints are multimodal; only the text model loads, and the sizes
above are for it. Fetch `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`model.safetensors.index.json` and every shard it names (see
`benchmarks/README.md` §0).

KV cache per 1000 tokens, for reference: Qwen3-8B is 36 layers x 8 KV heads x
128 dim x 2 (K+V) x 2 B = **147 MB/1k tokens** at bf16. A 32k-token context is
~4.7 GB — which is the regime where quantizing the cache finally matters, and
the whole reason for running this on a GPU rather than GPT-2 on a CPU.

---

## 3. Precision

On CPU the loader silently falls back to fp32, because **candle has no CPU bf16
gemm**. On CUDA bf16 works, so the same command produces a genuinely different
run:

```bash
./target/release/bench benchmarks/qwen3-0.6b 64 128
# CPU : "note: checkpoint is bf16, which Cpu cannot run; loading at f32 instead"
# CUDA: precision bf16, half the weight memory, no note
```

Pin it explicitly when comparing like with like:

```bash
./target/release/sweep ... --precision f32     # match the CPU numbers exactly
./target/release/sweep ... --precision bf16    # the realistic GPU deployment
```

Report both. fp32-on-GPU vs fp32-on-CPU isolates the hardware; bf16-vs-fp32 on
the same GPU isolates the precision.

---

## 4. The run sequence

Run these in order. Steps 1 and 2 are correctness gates — **if they fail, no
timing number below is worth recording.**

### 4.1 Parity vs HuggingFace (gate)

```bash
python -m pip install numpy torch transformers        # once

# GPT-2, both paths
./target/release/parity_dump --                      benchmarks/prompts.txt benchmarks/gpt2_logits.json  benchmarks/gpt2
python benchmarks/parity.py benchmarks/gpt2_logits.json
./target/release/parity_dump --cache                 benchmarks/prompts.txt benchmarks/gpt2_logits.json  benchmarks/gpt2
python benchmarks/parity.py benchmarks/gpt2_logits.json

# Qwen3, both paths
./target/release/parity_dump --                      benchmarks/prompts.txt benchmarks/qwen3_logits.json benchmarks/qwen3-0.6b
python benchmarks/parity.py benchmarks/qwen3_logits.json
./target/release/parity_dump --cache                 benchmarks/prompts.txt benchmarks/qwen3_logits.json benchmarks/qwen3-0.6b
python benchmarks/parity.py benchmarks/qwen3_logits.json
```

Pass criteria: `mean cos > 0.9999`, `top1 = 24/24`, `mean mse < 1e-3`.

> At **bf16** these thresholds do not apply — bf16 has ~3 decimal digits, so
> expect `mse ~1e-2` and occasional top-1 disagreement on near-ties. Run the
> parity gate at `--precision f32` on the GPU, then report bf16 quality through
> perplexity (§4.4) instead, which is the metric that survives low precision.

### 4.2 Cache correctness (gate)

```bash
cargo test --release --features cuda --test cache_parity -- --ignored --nocapture
```

All 13 must pass (each architecture's cases skip if its weights are absent).
`cache_matches_no_cache` is the important one: greedy decode
with the cache must produce byte-identical token ids to full recompute.

### 4.3 Throughput sweep

```bash
# GPT-2, matching the CPU table exactly
./target/release/sweep benchmarks/gpt2 16,64,128,256,512 32 both none \
  --repeats 5 --precision f32 > benchmarks/sweep_gpu_gpt2.csv

# Qwen3-0.6B at bf16, the realistic deployment
./target/release/sweep benchmarks/qwen3-0.6b 16,64,128,256,512,1024 32 both none \
  --repeats 5 --precision bf16 > benchmarks/sweep_gpu_qwen3.csv

# The quantization ablation: one run per mode
for q in none int8 int4; do
  ./target/release/sweep benchmarks/qwen3-0.6b 128,512,2048,8192 32 on $q \
    --repeats 5 --precision bf16 >> benchmarks/sweep_gpu_qwen3_quant.csv
done
```

Every row ends with `threads` and `cpu_simd`, the CPU side of the execution
plan. On a GPU run they describe only the host, but keep them: CPU rows from the
same machine are worth adding (build once without `--features cuda` into a
separate `--target-dir`), and the columns say which is which.

`--repeats 5` minimum for anything you plan to cite. A GPU is far less noisy
than a loaded CPU, but the first iteration still pays for lazy kernel selection;
the harness does one untimed warm-up per config.

**Push the context length.** The CPU tables stop at 512 tokens because that is
where CPU decode becomes unusable. On a 5090, Qwen3 can run to ~8k, and that
is where the KV cache stops being a rounding error against the weights and the
quantization ablation acquires a point. Long-context rows are the ones worth
having. The ceiling is prefill, not the cache: without a FlashAttention path the
prompt's full attention-score matrix is materialised, ~2 GB at 8k for
Qwen3-0.6B and ~34 GB at 32k.

**Qwen3 against Qwen3.5 at long context.** On the CPU, Qwen3.5-0.8B lost 8% of
its decode rate from 64 to 960 tokens and Qwen3-0.6B lost 27%, and the two were
within 4% at 960 (`benchmarks/README.md` §10). Where they cross is a GPU
question:

```bash
./target/release/sweep benchmarks/qwen3-0.6b,benchmarks/qwen3.5-0.8b \
  512,2048,8192 32 on none --repeats 5 --precision bf16 > benchmarks/sweep_gpu_hybrid.csv
```

Stop at 8k. Prefill materialises the full attention-score matrix (there is no
FlashAttention path): for Qwen3-0.6B at 32k tokens that is 16 heads × 32768² ×
2 B ≈ 34 GB, more than the card has. 8k is ~2 GB.

Expect Qwen3.5 prefill to be the slow column: its DeltaNet layers prefill token
by token, and on a GPU that is thousands of tiny kernel launches per layer. Time
the long rows with that in mind, and treat prefill and decode separately.

### 4.4 Perplexity / quality

```bash
# fetch WikiText-2 once (see benchmarks/README.md §5)
for q in off int8 int4; do
  ./target/release/ppl --kv-quant $q benchmarks/qwen3-0.6b benchmarks/wikitext2.txt 1024 512 60000
done
```

Report Δperplexity against the `off` baseline at a fixed `MAX_TOKENS`. This is
the quality axis of the ablation; the drift numbers in `cache_parity` are a
single-prompt smoke test, not a quality measurement.

> Expect Qwen3 to degrade **more** than GPT-2 under int8/int4 KV — see the
> architecture-sensitivity note in `benchmarks/README.md`. Confirming that on
> perplexity (not just one prompt's logits) is the single most interesting
> number this GPU run can produce.

### 4.5 Memory

```bash
for q in none int8 int4; do
  ./target/release/memprobe on 2048 32 benchmarks/qwen3-0.6b $q
done
./target/release/memprobe off 2048 32 benchmarks/qwen3-0.6b
```

`memprobe` reports host peak RSS, which on a GPU run is **not** the interesting
number — the cache lives in VRAM. Capture device memory alongside it:

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem.log &
./target/release/memprobe on 2048 32 benchmarks/qwen3-0.6b int8
kill %1
```

The CPU write-up notes that peak RSS never moved because a ~977 MiB load-time
floor dominated the ≤72 MiB cache. On a GPU at 8k+ context with an 8B model the
cache is multiple GB, so the retained saving should finally show up as a real
VRAM difference. **That is the measurement the CPU could not make**, and it is
worth stating plainly whether it materialises or not.

---

## 5. Recording results

For every table, record alongside the numbers:

- GPU model, driver version, CUDA toolkit version
- `rustc --version`, candle version (0.11.0)
- precision, `--repeats`, and mean ± sample std
- weight file SHA-256 (must match `benchmarks/README.md` §0)

Add GPU rows next to the CPU ones rather than replacing them — the CPU/GPU
contrast is a result, and overwriting it loses the comparison.

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `no kernel image is available` | CUDA toolkit too old for the GPU's compute capability (need 12.8+ for a 5090) |
| Throughput matches CPU | built without `--features cuda` |
| `unsupported dtype BF16 for op matmul` | running on CPU after all — check the plan line says `device=cuda` |
| the plan line says `half=f16` on an RTX 30/40/50 card | the bf16 probe failed; see §1.1 |
| `MatMulNonContiguous` in the cache gate | cuBLAS rejected the strided KV-cache view; a crate bug, see §1.1 |
| `CUDA out of memory` on a long-context row | KV cache: lower the prefill, or use `int8` |
| Parity fails only at bf16 | expected — run the parity gate at `--precision f32`, see §4.1 |
| Wildly variable first trial | lazy kernel selection; the harness warms up once per config, raise `--repeats` |
