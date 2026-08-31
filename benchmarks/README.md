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

The binaries also accept a Hub id (`openai-community/gpt2`) instead of the
directory, if you'd rather use the HF cache.

## 1. Throughput and size

```bash
cargo run --release --bin bench -- benchmarks/gpt2 64 128            # CPU
cargo run --release --features cuda --bin bench -- benchmarks/gpt2 64 128   # GPU
```

Arguments: `MODEL PREFILL_TOKENS DECODE_STEPS`.

Reports: load time, analytic parameter count, fp32 weight memory, prefill
latency (tok/s), and greedy decode throughput **both without and with the KV
cache**, plus the speedup. It does **not** measure peak RAM itself — wrap the
command:

| OS | command | field |
|---|---|---|
| Linux | `/usr/bin/time -v <cmd>` | `Maximum resident set size` |
| macOS | `/usr/bin/time -l <cmd>` | `maximum resident set size` |
| Windows | `Get-Process bench \| Select-Object PeakWorkingSet64` | bytes |

> `decode(no cache)` recomputes the whole sequence every step (O(n²)), so its
> tok/s falls as the sequence grows; `decode(KV cache)` prefills once then feeds
> one token per step (O(n)). Keep the same `PREFILL`/`DECODE` args when comparing
> across changes.

### Results — `openai-community/gpt2` (124M), fp32

Environment: AMD Ryzen 5 5600G (6C/12T), 28 GiB RAM, Windows 11 Pro ·
rustc 1.98.0 `x86_64-pc-windows-gnu` (MSYS2 mingw-w64 gcc 16.2.0) ·
candle 0.11.0 · `--release` · weights loaded from `benchmarks/gpt2/`.

| metric | CPU | GPU |
|---|---|---|
| hardware | Ryzen 5 5600G, 28 GiB | not measured (no CUDA toolkit on this box) |
| load time (local files) | 0.4–0.6 s | — |
| parameters | 124,439,808 | — |
| fp32 weights | ~0.46 GiB | — |
| prefill, 64 tok | **~366 tok/s** (360–420 across runs) | — |
| decode, 128 steps (seq 64→192), no cache | **~3 tok/s** (~40 s) | — |
| decode, 128 steps (seq 64→192), KV cache | **~43 tok/s** (~3 s) — **≈14× speedup** | — |
| peak RSS | **977 MiB** | — |

_GPU row: run the `--features cuda` command on a machine with the CUDA toolkit._
_The KV-cache speedup grows with sequence length — see the sweep in §4._

## 2. Output parity vs HuggingFace Transformers

Confirms MiniLLM computes the same thing as the reference implementation, up to
fp32 rounding.

```bash
python -m pip install numpy torch transformers        # once

cargo run --release --bin parity_dump                 # forward path  -> benchmarks/minillm_logits.json
cargo run --release --bin parity_dump -- --cache      # KV-cache path -> same file
python benchmarks/parity.py benchmarks/minillm_logits.json
```

`parity_dump` records the exact `input_ids` it used and the absolute path to the
weights directory; `parity.py` loads the HF reference from that same directory
with `local_files_only=True` and feeds it the same ids — so the check is
tokenizer-independent. It compares the raw logits for the position right after
the last prompt token. With `--cache`, MiniLLM prefills all but the last token
through the cache and feeds the last one as a single decode step, so the offset
mask and the single-token path are exercised.

Metrics: `mse`, `mae`, `max|Δ|`, cosine similarity,
`KL(softmax(hf) || softmax(mini))`, top-1 match, top-5 set overlap.

**Pass criteria** (fp32): `mean cos > 0.9999`, `top1 = n/n`, `mean mse ≲ 1e-3`.
A large KL or any top-1 miss means a real discrepancy (weight transpose,
activation, LayerNorm eps, position ids, tying).

### Results — `openai-community/gpt2`

Reference: transformers 5.16.1, torch 2.13.0+cpu, numpy 2.5.2 · same fp32
`benchmarks/gpt2/` weights.

| aggregate | `forward` | `--cache` |
|---|---|---|
| prompts | 24 | 24 |
| mean mse | **1.996e-09** | **1.912e-09** |
| mean cosine similarity | **1.00000** | **1.00000** |
| mean KL(hf‖mini) | **3.984e-10** | **1.647e-10** |
| per-prompt max\|Δ\| (logit) | ~1–2 × 10⁻⁴ | ~1–2 × 10⁻⁴ |
| top-1 agreement | **24 / 24** | **24 / 24** |
| top-5 overlap | **120 / 120** | **120 / 120** |

Both the full-recompute forward pass and the KV-cache decode path match
HuggingFace to floating-point noise. The Rust integration test
`tests/cache_parity.rs` additionally asserts that greedy generation with the
cache yields the exact same token ids as `forward` (run:
`cargo test --test cache_parity -- --ignored`).

## 3. Larger GPT-2 variants (optional)

`bench` and `parity_dump` accept `openai-community/gpt2-medium`, `-large`,
`-xl` (or a local dir for each). Record a row per size once weight loading for
those is verified.

| model | params | load (s) | prefill tok/s | decode tok/s (cache) | peak RSS | mean cos vs HF |
|---|---|---|---|---|---|---|
| gpt2 | 124M | 0.4–0.6 | ~366 | ~43 (seq 64→192) | 977 MiB | 1.00000 |
| gpt2-medium | 355M | | | | | |
| gpt2-large | 774M | | | | | |

## 4. Sequence-length sweep — KV cache on vs off (`sweep`)

Sweeps a set of models against a set of prefill lengths, runs a greedy decode
loop after each with the cache **off** and **on**, and writes one CSV row per
`(model, prefill, kv_cache)` to stdout.

```bash
cargo run --release --bin sweep -- benchmarks/gpt2 16,64,128,256,512 32 both > benchmarks/sweep_cpu.csv
# args: MODELS(comma-sep dirs/ids)  PREFILLS(comma-sep)  DECODE(steps)  CACHE(off|on|both)
```

Columns: `model,params,kv_bytes_per_token,device,dtype,kv_cache,kv_quant,prefill_tokens,decode_steps,seq_start,seq_end,load_s,prefill_ms,prefill_tok_s,decode_tok_s,decode_s`.
`kv_bytes_per_token` is the analytic retained cache cost per position for that
row's `kv_quant` (§6); `kv_quant` is `none` on `off` rows.

### Results — `openai-community/gpt2` (124M), fp32, CPU

Same environment as §1. 32 greedy decode steps after each prefill;
`benchmarks/sweep_cpu.csv` has the raw rows.

| seq start | decode tok/s, no cache | decode tok/s, KV cache | speedup |
|---|---|---|---|
| 16 | 11.3 | 52.1 | 4.6× |
| 64 | 5.1 | 49.0 | 9.7× |
| 128 | 2.9 | 45.9 | 15.7× |
| 256 | 1.6 | 37.8 | 23.5× |
| 512 | 0.8 | 25.2 | 31.9× |

No-cache decode ~halves per doubling of the sequence (O(n²)); the cached path
degrades gently (growing K/V matmul + the per-step `cat` copy), so the speedup
widens with context length.

## 5. Perplexity (`ppl`)

Sliding-window LM perplexity, the quality axis for the KV-cache-quantization
ablation. Each target token is scored exactly once, by the window with the most
left context; NLL is `logsumexp(row) - row[target]`, accumulated on the CPU.

```bash
# one-time: fetch the WikiText-2 raw test split into benchmarks/wikitext2.txt (git-ignored)
python -m pip install pyarrow
python - <<'PY'
import pyarrow.parquet as pq, urllib.request
u = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
urllib.request.urlretrieve(u, "wt2.parquet")
txt = "".join(pq.read_table("wt2.parquet").column("text").to_pylist())
open("benchmarks/wikitext2.txt", "w", encoding="utf-8", newline="").write(txt)
PY

cargo run --release --bin ppl -- benchmarks/gpt2 benchmarks/wikitext2.txt 512 256 8192
# args: MODEL  TEXT_FILE  WINDOW  STRIDE  MAX_TOKENS
```

### Correctness — vs HuggingFace Transformers

`ppl_ref.py` runs the identical windowing scheme on the same text file with HF as
the model (the `ppl` analogue of `parity.py`):

```bash
python benchmarks/ppl_ref.py benchmarks/gpt2 benchmarks/wikitext2.txt 512 256 8192
```

| first 8192 tokens, `WINDOW=512 STRIDE=256` | mean NLL (nats/tok) | perplexity |
|---|---|---|
| MiniLLM `ppl` | 3.51105 | 33.483 |
| HF Transformers `ppl_ref.py` | 3.51119 | 33.488 |

Agreement to ~1e-4 nats (fp32 noise) — the harness is correct.

### Baseline — `openai-community/gpt2` (124M), fp32

| tokens scored | `WINDOW`/`STRIDE` | mean NLL | perplexity |
|---|---|---|---|
| 8,191 (`MAX_TOKENS=8192`) | 512 / 256 | 3.51105 | **33.48** |
| 59,999 (`MAX_TOKENS=60000`) | 512 / 256 | 3.39918 | **29.94** |

The first ~8k tokens of the test split are short biographical stubs and score
high; over 60k it settles to ~29.9, in line with the usual "GPT-2 small ≈ 29" on
WikiText-2. The KV-cache-quant ablation reports Δperplexity against this same
fp32 baseline at a fixed `MAX_TOKENS`.

## 6. KV-cache memory (`memprobe`)

**Analytic** retained cost per cached position. fp32 is
`2 (K + V) * n_layer * n_embd * 4`; the int modes store the values narrower and
add one fp32 scale per `(layer, head, position, K|V)`
(`KvQuant::bytes_per_token`). For `gpt2` 124M:

| storage | bytes/token | full 1024-ctx cache | vs fp32 |
|---|---|---|---|
| fp32 | 73,728 (72 KiB) | 72 MiB | 1.00× |
| int8 | 19,584 (19 KiB) | 19 MiB | 3.76× smaller |
| int4 *(planned)* | 10,368 (10 KiB) | 10 MiB | 7.1× smaller |

**Measured** peak RSS, one decode path per process, `decode = 32`:

```bash
cargo run --release --bin memprobe -- off 992 32            # full-recompute path
cargo run --release --bin memprobe -- on  992 32 benchmarks/gpt2 none   # fp32 cache
cargo run --release --bin memprobe -- on  992 32 benchmarks/gpt2 int8   # int8 cache
```

| seq end | analytic KV (fp32 / int8) | peak RSS: no cache / fp32 cache / int8 cache |
|---|---|---|
| 128  | 9 / 2 MiB   | 977 / 977 / 977 MiB |
| 512  | 36 / 10 MiB | 977 / 977 / 977 MiB |
| 1024 | 72 / 19 MiB | 977 / 977 / 977 MiB |

Peak RSS does not move — with any cache setting. Two reasons: (1) the cache
(≤72 MiB fp32) is smaller than the activation transient one prefill forward
already allocates (`[1, seq, vocab]` fp32 logits alone are ~200 MiB at seq 1024),
so peak stays pinned at the ~977 MiB weights+transient floor; (2) the int8 path
still dequantizes the whole cache to fp32 for the attention matmul, so its
transient footprint matches the fp32 cache. The **retained** footprint is what
int8 shrinks (3.76×) — visible in the analytic column, not in peak RSS on this
config. Realising a peak-RSS win needs int8 matmul or chunked dequant, or a
regime where the cache is the dominant allocation (bigger model, longer context,
batch > 1).

## 7. KV-cache quantization ablation — int8

`ppl`, `sweep` and `memprobe` all take a quant mode (`--kv-quant int8` for `ppl`;
a `KVQUANT` positional for the others). `int8` stores each cached position's
per-head K/V vector as `u8` with one fp32 scale — per-token symmetric int8,
`x ≈ (q - 128) * scale` — and dequantizes the whole cache to fp32 for the
attention matmul. The current token is quantized like every other.

```bash
cargo run --release --bin ppl -- --kv-quant int8 benchmarks/gpt2 benchmarks/wikitext2.txt 512 256 60000
cargo run --release --bin sweep -- benchmarks/gpt2 16,64,128,256,512 32 on int8
```

**Quality** — WikiText-2 raw test, `WINDOW=512 STRIDE=256`:

| tokens scored | fp32 cache ppl | int8 cache ppl | Δppl | ΔNLL (nats/tok) |
|---|---|---|---|---|
| 8,191 | 33.4833 | 33.4849 | +0.0016 | +0.00004 |
| 59,999 | 29.9394 | 29.9475 | +0.0081 | +0.00027 |

Per-token int8 KV is effectively lossless — Δ is at the level of fp32 rounding.

**Memory** — retained cache shrinks 3.76× (72 → 19 KiB/token; §6). Peak RSS is
unchanged on this config (the attention path dequantizes to fp32).

**Speed** — decode throughput, `gpt2` 124M / CPU, 32 steps
(`benchmarks/sweep_cpu.csv`):

| seq start | fp32 cache | int8 cache | int8 / fp32 |
|---|---|---|---|
| 16  | 52.1 | 47.7 | 0.92× |
| 64  | 49.0 | 47.2 | 0.96× |
| 128 | 45.9 | 40.4 | 0.88× |
| 256 | 37.8 | 33.8 | 0.89× |
| 512 | 25.2 | 17.9 | 0.71× |

int8 decode runs at ~0.9× the fp32-cache rate up to ~256 tokens and ~0.7× by
512 — the per-step quantize + full-cache dequantize is overhead on a path that
still does the matmul in fp32, and it grows with the cache length. It stays far
above the no-cache rate. The trade: **3.76× smaller retained cache at ~0 quality
loss, for ~10–30% slower decode** (widening with context) until the matmul
itself goes int8.
