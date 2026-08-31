# Benchmarks

Reproducible throughput, memory and correctness checks for MiniLLM.

Everything here is deterministic given a model id and the fixed prompt set in
[`prompts.txt`](prompts.txt). Re-run after any change to the model or generation
code and paste the new numbers into the tables below.

## 1. Throughput and size

```bash
# CPU
cargo run --release --bin bench -- openai-community/gpt2 64 128

# GPU
cargo run --release --features cuda --bin bench -- openai-community/gpt2 64 128
```

Arguments: `MODEL_ID PREFILL_TOKENS DECODE_STEPS`.

The harness reports: model load time, analytic parameter count, fp32 weight
memory, prefill latency (tok/s), and greedy decode throughput (tok/s). It does
**not** measure peak RAM itself — wrap the command:

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

| metric | CPU | GPU |
|---|---|---|
| hardware | _fill in (CPU model, RAM)_ | _fill in (GPU model, VRAM)_ |
| candle / rustc version | _fill in_ | _fill in_ |
| load time (s) | _run_ | _run_ |
| parameters | 124,439,808 | 124,439,808 |
| fp32 weights (GiB) | ~0.46 | ~0.46 |
| prefill 64 tok (tok/s) | _run_ | _run_ |
| decode 128 steps (tok/s) | _run_ | _run_ |
| peak RSS (MiB) | _measure externally_ | _measure externally_ |

_(Run the commands above and replace the `_run_` cells. Do not estimate — the
point of this file is measured numbers.)_

## 2. Output parity vs HuggingFace Transformers

Confirms MiniLLM computes the same thing as the reference implementation, up to
fp32 rounding.

```bash
# 1. dump MiniLLM logits for the fixed prompts
cargo run --release --bin parity_dump          # -> benchmarks/minillm_logits.json
#    (add --features cuda to dump from the GPU path)

# 2. compare against transformers
python -m pip install numpy torch transformers   # once
python benchmarks/parity.py benchmarks/minillm_logits.json
```

`parity_dump` records the exact `input_ids` it used, so `parity.py` feeds
HuggingFace the same ids — the check is tokenizer-independent. It compares the
raw logits for the position right after the last prompt token.

Metrics: `mse`, `mae`, `max|Δ|`, cosine similarity, `KL(softmax(hf) || softmax(mini))`,
top-1 match, top-5 set overlap.

**Pass criteria** (fp32): `mean cos > 0.9999`, `top1 = n/n`, `mean mse` at or
below ~`1e-3`. A large KL or any top-1 miss means a real discrepancy (weight
transpose, activation, LayerNorm eps, position ids, tying).

### Results — `openai-community/gpt2`

| aggregate | value |
|---|---|
| prompts | 24 |
| mean mse | _run_ |
| mean cosine similarity | _run_ |
| mean KL(hf‖mini) | _run_ |
| top-1 agreement | _run_ / 24 |
| top-5 overlap | _run_ / 120 |
| transformers version | _fill in_ |

## 3. Larger GPT-2 variants (optional)

The same commands accept `openai-community/gpt2-medium`, `-large`, `-xl`. Record
a row per size once weight loading for those is verified.

| model | params | load (s) | prefill tok/s | decode tok/s | peak RSS | mean cos vs HF |
|---|---|---|---|---|---|---|
| gpt2 | 124M | | | | | |
| gpt2-medium | 355M | | | | | |
| gpt2-large | 774M | | | | | |
