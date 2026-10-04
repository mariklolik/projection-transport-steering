from __future__ import annotations

import argparse
import json

import torch

from behaviour_specific.overconfidence.label_pool import LAYERS, prompt_features
from general.inference import generate_batch_chunked, get_trace_activations
from general.paths import DATA_DIR, RESULTS_DIR
from general.storage import write_jsonl
from models_specific.active import chat_prompt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--outdir", default="v4_oodneg")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from models_specific.active import load_model

    held = set(json.loads((DATA_DIR / "openended_prompts.json").read_text()))
    src = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    pool = [r["instruction"] for r in json.loads(open(src).read())]
    texts = [t for t in dict.fromkeys(pool) if t not in held][: args.n]
    model, tok = load_model()
    prompts = [chat_prompt(tok, t) for t in texts]
    outs = generate_batch_chunked(model, tok, prompts, batch_size=64)
    rows = [{"id": f"ood-{i}", "state": "out_of_domain", "generations": [{"prompt": p, "text": o}]}
            for i, (p, o) in enumerate(zip(prompts, outs))]
    out = RESULTS_DIR / args.outdir
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rollouts" / "ood__shard0.jsonl", rows)
    torch.save({"ids": [r["id"] for r in rows], "layers": LAYERS,
                "X": torch.stack([get_trace_activations(model, tok, p, o)[list(LAYERS)].half() for p, o in zip(prompts, outs)])},
               out / "feats__shard0.pt")
    torch.save({"ids": [r["id"] for r in rows], "layers": LAYERS, "pos": ("last", "mean"),
                "X": prompt_features(model, tok, rows)}, out / "prompt__shard0.pt")
    print(len(rows), "->", out)
