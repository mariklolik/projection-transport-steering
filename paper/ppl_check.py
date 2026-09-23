from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

STRIDE, WINDOW = 512, 1024


@torch.no_grad()
def corpus_ppl(model, tok, text: str) -> tuple[float, int]:
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    nll, n = 0.0, 0
    for i in range(0, ids.shape[1], STRIDE):
        chunk = ids[:, i:i + WINDOW]
        if chunk.shape[1] < 16:
            break
        out = model(chunk, labels=chunk)
        t = chunk.shape[1] - 1
        nll += float(out.loss) * t
        n += t
    return math.exp(nll / max(n, 1)), n


@torch.no_grad()
def sent_ppl(model, tok, text: str) -> float:
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    if ids.shape[1] < 4:
        return float("nan")
    return math.exp(float(model(ids, labels=ids).loss))


def _selftest():
    assert abs(math.exp(math.log(7.0)) - 7.0) < 1e-9
    print("ppl_check self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prose_dir")
    ap.add_argument("--model", default="gemma_2_2b_it")
    ap.add_argument("--per-sentence", default="", help="stem to break down line by line")
    args = ap.parse_args()

    _selftest()

    import os
    os.environ["MODEL_IMPL"] = args.model
    from models_specific.active import load_model

    model, tok = load_model()
    rows = {}
    for p in sorted(Path(args.prose_dir).glob("*.txt")):
        ppl, n = corpus_ppl(model, tok, p.read_text())
        rows[p.stem] = (ppl, n)
        print(f"  {p.stem:14s} ppl {ppl:8.3f}  ({n} tokens)", flush=True)

    if args.per_sentence:
        lines = (Path(args.prose_dir) / f"{args.per_sentence}.txt").read_text().splitlines()
        scored = sorted(((sent_ppl(model, tok, x), x) for x in lines if x.strip()),
                        key=lambda t: -t[0])
        print("\nhighest-perplexity sentences:")
        for v, x in scored[:15]:
            print(f"  {v:8.1f}  {x[:110]}")
        print("\nlowest-perplexity sentences:")
        for v, x in scored[-5:]:
            print(f"  {v:8.1f}  {x[:110]}")

    refs = [v[0] for k, v in rows.items() if k != "ours"]
    if refs and "ours" in rows:
        lo, hi = min(refs), max(refs)
        mean = sum(refs) / len(refs)
        o = rows["ours"][0]
        print(f"\nreference band {lo:.3f}-{hi:.3f} (mean {mean:.3f}); ours {o:.3f}, "
              f"{'inside' if lo <= o <= hi else 'outside'}; "
              f"relative gap to mean {100 * (o - mean) / mean:+.1f}%")
