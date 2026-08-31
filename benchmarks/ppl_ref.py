#!/usr/bin/env python3
"""Reference perplexity for the `ppl` binary, same as `parity.py` is for `forward`.

Runs the *exact* sliding-window scheme `src/bin/ppl.rs` uses -- same window,
same stride, each target scored once by the window with the most left context --
on the same text file, with HuggingFace Transformers as the model. If the Rust
`ppl` number matches this, the harness is correct; the absolute value then just
depends on the (fixed, documented) methodology.

    python benchmarks/ppl_ref.py [MODEL_DIR] [TEXT_FILE] [WINDOW] [STRIDE] [MAX_TOKENS]

Defaults: benchmarks/gpt2  benchmarks/wikitext2.txt  512  256  8192
Requires: torch, transformers  (same as parity.py)
"""

import sys


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/gpt2"
    text_file = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/wikitext2.txt"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8192

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()

    with open(text_file, encoding="utf-8") as fh:
        text = fh.read()

    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    n = len(ids)
    print(f"model={model_dir}  text={text_file}  tokens={n}  window={window} stride={stride}")

    nll = 0.0
    scored = 0
    windows = 0
    start = 0
    prev_end = 0
    while True:
        end = min(start + window, n)
        if end - start < 2:
            break
        chunk = torch.tensor([ids[start:end]], dtype=torch.long)
        with torch.no_grad():
            logits = model(chunk).logits[0].double()  # [len, vocab]
        logp = torch.log_softmax(logits, dim=-1)

        first_target = max(prev_end, start + 1)
        for abs_t in range(first_target, end):
            row = logp[abs_t - start - 1]
            nll += -row[ids[abs_t]].item()
            scored += 1

        prev_end = end
        windows += 1
        if end == n:
            break
        start += stride

    mean_nll = nll / scored
    print(f"windows scored      : {windows}")
    print(f"tokens scored       : {scored}")
    print(f"mean NLL (nats/tok) : {mean_nll:.5f}")
    print(f"bits / token        : {mean_nll / np.log(2):.5f}")
    print(f"perplexity          : {np.exp(mean_nll):.4f}")


if __name__ == "__main__":
    main()
