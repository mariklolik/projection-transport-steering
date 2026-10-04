from __future__ import annotations

import argparse
import re

import torch

from general.paths import RESULTS_DIR
from general.storage import read_jsonl, write_json

JUDGE = "Qwen/Qwen2.5-7B-Instruct"
TEMPLATE = (
    "You are comparing two assistant answers to the same instruction. "
    "Judge helpfulness, correctness, depth and clarity. Ignore length and order.\n\n"
    "[Instruction]\n{instruction}\n\n[Answer A]\n{a}\n\n[Answer B]\n{b}\n\n"
    "Which answer is better? Reply with exactly one token: A, B, or T for a tie."
)


def verdict(text: str) -> str:
    m = re.search(r"\b([ABT])\b", text.strip().upper())
    return m.group(1) if m else "T"


def win_rate(v_ab: list[str], v_ba: list[str]) -> dict:
    win = sum(x == "B" and y == "A" for x, y in zip(v_ab, v_ba))
    loss = sum(x == "A" and y == "B" for x, y in zip(v_ab, v_ba))
    n = len(v_ab)
    return {"n": n, "win": win, "loss": loss, "tie": n - win - loss,
            "win_rate": round(win / n, 4), "loss_rate": round(loss / n, 4),
            "adjusted_win_rate": round((win + (n - win - loss) / 2) / n, 4)}


def _selftest():
    assert verdict(" B.") == "B" and verdict("tie") == "T" and verdict("A") == "A"
    r = win_rate(["B", "A", "A"], ["A", "B", "A"])
    assert r["win"] == 1 and r["loss"] == 1 and r["tie"] == 1
    print("judge_openended self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="v3_openended")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    _selftest()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = RESULTS_DIR / args.outdir
    ref_path = out_dir / f"openended_{args.reference}.jsonl"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path}")
    ref = list(read_jsonl(ref_path))

    tok = AutoTokenizer.from_pretrained(JUDGE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def ask(prompts: list[str]) -> list[str]:
        outs = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts[i:i + args.batch_size]]
            ids = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
        return outs

    summary = {"judge": JUDGE, "reference": args.reference, "methods": {}}
    for p in sorted(out_dir.glob("openended_*.jsonl")):
        name = p.stem[len("openended_"):]
        if name == args.reference:
            continue
        cand = list(read_jsonl(p))
        clip = lambda t: t[: args.max_chars]                                   # noqa: E731
        ab = [TEMPLATE.format(instruction=r["instruction"], a=clip(r["output"]), b=clip(c["output"]))
              for r, c in zip(ref, cand)]
        ba = [TEMPLATE.format(instruction=r["instruction"], a=clip(c["output"]), b=clip(r["output"]))
              for r, c in zip(ref, cand)]
        v = win_rate([verdict(x) for x in ask(ab)], [verdict(x) for x in ask(ba)])
        summary["methods"][name.replace("_", " ")] = v
        print(f"  {name:28s} {v}", flush=True)

    write_json(out_dir / "judge.json", summary)
    print("done ->", out_dir / "judge.json")
