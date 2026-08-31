#!/usr/bin/env python3
"""Compare MiniLLM logits against HuggingFace Transformers.

Pipeline:
    1. cargo run --release --bin parity_dump          # writes minillm_logits.json
    2. python benchmarks/parity.py benchmarks/minillm_logits.json

The Rust side records `input_ids`, so this script feeds HuggingFace the exact
same token ids -- the comparison is independent of any tokenizer differences.

Metrics per prompt:
    mse        mean squared error between raw logit vectors
    mae        mean absolute error
    max|Δ|     largest absolute logit difference
    cos        cosine similarity of the logit vectors
    KL(hf||mi) KL divergence of softmax(hf) from softmax(mini), in nats
    top1       1 if argmax matches, else 0
    top5       size of the intersection of the two top-5 token sets (0..5)

Requires: numpy, torch, transformers
    pip install numpy torch transformers
"""

import json
import sys


def softmax(x):
    import numpy as np

    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl(p, q, eps=1e-9):
    import numpy as np

    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/minillm_logits.json"
    with open(path) as fh:
        data = json.load(fh)

    try:
        import numpy as np
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"missing dependency: {exc}\n  pip install numpy torch transformers")

    model_id = data.get("model", "openai-community/gpt2")
    print(f"reference: {model_id} (HuggingFace Transformers)")
    model = GPT2LMHeadModel.from_pretrained(model_id).eval()

    header = (
        f"{'prompt':<34}{'mse':>11}{'mae':>9}{'max|d|':>9}"
        f"{'cos':>10}{'KL(hf||mi)':>13}{'top1':>6}{'top5':>6}"
    )
    print(header)
    print("-" * len(header))

    n = len(data["entries"])
    sum_mse = sum_cos = sum_kl = 0.0
    top1_hits = 0
    top5_total = 0

    for ent in data["entries"]:
        ids = torch.tensor([ent["input_ids"]], dtype=torch.long)
        with torch.no_grad():
            hf = model(ids).logits[0, -1, :].double().numpy()
        mi = np.asarray(ent["logits"], dtype=np.float64)

        if mi.shape != hf.shape:
            sys.exit(
                f"shape mismatch for prompt {ent['prompt']!r}: "
                f"minillm {mi.shape} vs hf {hf.shape}"
            )

        diff = mi - hf
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        maxd = float(np.max(np.abs(diff)))
        cos = float(np.dot(mi, hf) / (np.linalg.norm(mi) * np.linalg.norm(hf)))
        kldiv = kl(softmax(hf), softmax(mi))
        t1 = int(np.argmax(hf) == np.argmax(mi))
        t5 = len(set(np.argsort(hf)[-5:].tolist()) & set(np.argsort(mi)[-5:].tolist()))

        sum_mse += mse
        sum_cos += cos
        sum_kl += kldiv
        top1_hits += t1
        top5_total += t5

        label = ent["prompt"][:32]
        print(
            f"{label:<34}{mse:>11.3e}{mae:>9.4f}{maxd:>9.4f}"
            f"{cos:>10.5f}{kldiv:>13.3e}{t1:>6}{t5:>6}"
        )

    print("-" * len(header))
    print(
        f"mean mse={sum_mse / n:.3e}   mean cos={sum_cos / n:.5f}   "
        f"mean KL={sum_kl / n:.3e}   top1={top1_hits}/{n}   "
        f"top5 overlap={top5_total}/{5 * n}"
    )
    print(
        "\nrule of thumb: fp32 numerical noise gives mse ~1e-3 or below, "
        "cos > 0.9999, top1 = n/n. Large KL or top1 misses => a real bug."
    )


if __name__ == "__main__":
    main()
