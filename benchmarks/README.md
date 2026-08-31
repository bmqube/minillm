# Benchmarks

Reproducible throughput, memory and correctness checks for MiniLLM.

Every benchmark here decodes **greedily**, which consumes no randomness, so the
outputs are bit-deterministic given the weights (hashes below) and the fixed
prompt set in [`prompts.txt`](prompts.txt). Only *timings* vary between runs —
see the note on repeats in §4. For reproducible *sampled* generation, use
`generation::sample_with` / `Generator::next_token_with` with a seeded RNG.

Re-run after any change to the model or generation code and update the tables
below.

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

Verify you have the same bytes every result below was produced from:

```bash
sha256sum benchmarks/gpt2/*            # or: shasum -a 256 / Get-FileHash -Algorithm SHA256
```

| file | SHA-256 |
|---|---|
| `config.json` | `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4` |
| `tokenizer.json` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `model.safetensors` | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| `wikitext2.txt` (§5) | `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a` |

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

Mean ± sample std over **5 trials** per configuration (`--repeats 5`);
`benchmarks/sweep_cpu.csv` has all 100 individual trials.

| seq start | decode tok/s, no cache | decode tok/s, KV cache | speedup |
|---|---|---|---|
| 16  | 11.76 ± 0.28 | 53.05 ± 1.08 | 4.5× |
| 64  | 5.19 ± 0.05 | 49.95 ± 0.60 | 9.6× |
| 128 | 2.96 ± 0.09 | 45.41 ± 1.36 | 15.3× |
| 256 | 1.64 ± 0.01 | 40.91 ± 1.34 | 24.9× |
| 512 | 0.80 ± 0.01 | 25.83 ± 0.14 | 32.3× |

No-cache decode ~halves per doubling of the sequence (O(n²)); the cached path
degrades gently (growing K/V matmul + the per-step `cat` copy), so the speedup
widens with context length.

Run-to-run spread is 0.4–4% on an otherwise-idle box but can exceed 10% under
load, so single-trial numbers are not trustworthy — always pass `--repeats`.

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
| int4 | 11,520 (11 KiB) | 11 MiB | 6.40× smaller |

int8 adds one fp32 scale per `(layer, head, position, K|V)`; int4 packs two
values per byte and adds a scale **and** a zero-point.

**Measured** peak RSS, one decode path per process, `decode = 32`:

```bash
cargo run --release --bin memprobe -- off 992 32            # full-recompute path
cargo run --release --bin memprobe -- on  992 32 benchmarks/gpt2 none   # fp32 cache
cargo run --release --bin memprobe -- on  992 32 benchmarks/gpt2 int8   # int8 cache
cargo run --release --bin memprobe -- on  992 32 benchmarks/gpt2 int4   # int4 cache
```

| seq end | analytic KV (fp32 / int8 / int4) | peak RSS (no cache / fp32 / int8 / int4) |
|---|---|---|
| 128  | 9 / 2 / 1 MiB   | 977 / 977 / 977 / 977 MiB |
| 512  | 36 / 10 / 6 MiB | 977 / 977 / 977 / 977 MiB |
| 1024 | 72 / 19 / 11 MiB | 977 / 977 / 977 / 977 MiB |

Peak RSS does not move — with any cache setting. Two reasons: (1) the cache
(≤72 MiB fp32) is smaller than the activation transient one prefill forward
already allocates (`[1, seq, vocab]` fp32 logits alone are ~200 MiB at seq 1024),
so peak stays pinned at the ~977 MiB weights+transient floor; (2) every quantized
path still dequantizes the whole cache to fp32 for the attention matmul, so its
transient footprint matches the fp32 cache. The **retained** footprint is what
quantization shrinks — visible in the analytic column, not in peak RSS on this
config. Realising a peak-RSS win needs a low-bit matmul or chunked dequant, or a
regime where the cache is the dominant allocation (bigger model, longer context,
batch > 1).

## 7. KV-cache quantization ablation — int8 & int4

`ppl`, `sweep` and `memprobe` take a quant mode (`--kv-quant` for `ppl`; a
`KVQUANT` positional for the others). Both modes quantize **per token**: one
group per `(batch, head, position)` covering that position's `head_dim` vector,
and every token — including the newest — is quantized. Storage is dequantized to
fp32 for the attention matmul.

- **int8** — symmetric, `u8` + fp32 scale, `x ≈ (q - 128) * scale`.
- **int4** — asymmetric, two `0..15` nibbles packed per `u8`, + fp32 scale and
  zero-point, `x ≈ q * scale + zero`.

```bash
cargo run --release --bin ppl -- --kv-quant int8 benchmarks/gpt2 benchmarks/wikitext2.txt 512 256 60000
cargo run --release --bin sweep -- benchmarks/gpt2 16,64,128,256,512 32 on int4
```

### Quality — WikiText-2 raw test, `WINDOW=512 STRIDE=256`

| tokens | fp32 ppl | int8 ppl (Δ) | int4 ppl (Δ) |
|---|---|---|---|
| 8,191  | 33.4833 | 33.4849 (**+0.0016**) | 34.6856 (**+1.20**, +3.6%) |
| 59,999 | 29.9394 | 29.9475 (**+0.0081**) | 31.0665 (**+1.13**, +3.8%) |

int8 is effectively lossless (Δ at fp32-rounding level). int4 is the first point
where quality actually breaks — ~+3.6% perplexity, from the per-token asymmetric
scheme with no full-precision residual for recent tokens.

### Memory — retained cache (§6)

| | fp32 | int8 | int4 |
|---|---|---|---|
| bytes/token | 73,728 | 19,584 (3.76× smaller) | 11,520 (6.40× smaller) |
| peak RSS | — unchanged (fp32 dequant transient dominates) — | | |

### Speed — decode tok/s, `gpt2` 124M / CPU, 32 steps (`benchmarks/sweep_cpu.csv`)

Mean ± std over 5 trials, same runs as §4.

| seq start | fp32 | int8 | int4 |
|---|---|---|---|
| 16  | 53.05 ± 1.08 | 51.69 ± 0.55 (0.97×) | 43.35 ± 0.17 (0.82×) |
| 64  | 49.95 ± 0.60 | 48.54 ± 0.36 (0.97×) | 33.89 ± 0.85 (0.68×) |
| 128 | 45.41 ± 1.36 | 44.23 ± 1.35 (0.97×) | 26.38 ± 0.22 (0.58×) |
| 256 | 40.91 ± 1.34 | 37.70 ± 1.49 (0.92×) | 17.13 ± 1.96 (0.42×) |
| 512 | 25.83 ± 0.14 | 19.90 ± 0.13 (0.77×) | 9.64 ± 0.21 (0.37×) |

int8 costs ~3% up to 128 tokens and 23% by 512; int4 starts at ~18% and reaches
63%. Both still beat the no-cache rate (0.8–11.8 tok/s over this range) by a wide
margin.

> **Why the quantized paths get slower with context — read this before citing the
> speed numbers.** Two properties of `LayerKvCache` drive them, and both are
> implementation choices, not properties of low-bit KV:
>
> 1. `append` grows the cache with `Tensor::cat`, copying the whole cache every
>    decode step. Applies to fp32 too, and is why even the fp32 cached path
>    decays with length.
> 2. Quantized modes **dequantize the entire cache to fp32 every step** so the
>    attention matmul stays fp32 — a second O(n) pass per step, heavier for int4
>    (nibble unpack) than int8 (byte cast).
>
> So these numbers bound how a *naive* low-bit KV path performs, not how int4
> must perform. A preallocated write-in-place buffer removes (1); a low-precision
> matmul or chunked dequant removes (2) and would also turn the retained-memory
> saving into a peak-RSS saving.

### The frontier

| mode | Δppl (60k) | retained KV | decode @ seq 128 |
|---|---|---|---|
| fp32 | — | 72 KiB/tok | 45.41 ± 1.36 tok/s |
| **int8** | +0.008 (+0.03%) | 19 KiB/tok (3.8×) | 44.23 ± 1.35 (0.97×) |
| **int4** | +1.13 (+3.8%) | 11 KiB/tok (6.4×) | 26.38 ± 0.22 (0.58×) |

int8 is essentially a free 3.8× memory cut — the quality change is at fp32-noise
level and the speed cost is inside the error bars at this context length. int4
buys another 1.7× compression for a real ~3.8% perplexity cost *and* a 42% speed
cost. Realising the memory saving as lower *peak* RSS, and recovering the speed,
both need the attention matmul itself to run at low precision.
