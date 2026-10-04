from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from behaviour_specific.overconfidence.eval_openended import WORD, distinct3, extract_code, run_humaneval_case
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from behaviour_specific.overconfidence.label_pool import extraction_records
from behaviour_specific.overconfidence.steer_overconfidence import mean_activation_norm
from general.inference import generate_batch_chunked, get_activations_all_layers, get_trace_activations
from general.paths import DATA_DIR, RESULTS_DIR
from general.steering import steer_add, steering_hook
from general.storage import write_json, write_jsonl
from models_specific.active import chat_prompt


@torch.no_grad()
def nll_each(model, tok, texts: list[str], max_len: int = 512) -> list[float]:
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        out.append(float(model(**ids, labels=ids["input_ids"]).loss))
    return out


def boot(x: np.ndarray, iters: int = 10000) -> list[float]:
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), (iters, len(x)))].mean(1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="humaneval")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alphas", default="-0.375,-0.5")
    ap.add_argument("--detector", default="detector_m5")
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--outdir", default="v4_capability")
    args = ap.parse_args()

    from datasets import load_dataset
    from models_specific.active import load_model

    model, tok = load_model()
    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    v = pts["dirs"]["m4_conf"].to(model.device, torch.float32)
    norm = mean_activation_norm(model, tok, extraction_records()[:300], args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    det = torch.load(DIRECTIONS_DIR / f"{args.detector}.pt")
    pdet = torch.load(DIRECTIONS_DIR / f"{args.detector}_prompt.pt")
    tau = float(np.interp(args.q, np.linspace(0.01, 0.99, 41), det["score_quantiles"].numpy()))
    ptau = float(np.interp(args.q, np.linspace(0.01, 0.99, 41), pdet["score_quantiles"].numpy()))

    if args.task == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t) > 400][:300]
        prompts, items = texts, texts
    elif args.task == "humaneval":
        rows = list(load_dataset("openai/openai_humaneval", split="test"))
        instr = ("Complete the following Python function. Reply with the full function "
                 "in a single ```python code block and nothing else.\n\n")
        prompts, items = [chat_prompt(tok, instr + r["prompt"]) for r in rows], rows
    else:
        items = json.loads((DATA_DIR / "openended_prompts.json").read_text())[:200]
        prompts = [chat_prompt(tok, t) for t in items]

    def hooked(alpha: float | None):
        return steering_hook(model.model.layers[args.layer], lambda x, a=alpha: steer_add(x, v, a * norm)) if alpha else None

    def run(alpha: float | None) -> list[str]:
        h = hooked(alpha)
        try:
            return generate_batch_chunked(model, tok, prompts, batch_size=args.batch_size,
                                          max_new_tokens=args.max_new_tokens)
        finally:
            if h:
                h.remove()

    def per_item(alpha: float | None, outs: list[str] | None) -> dict[str, np.ndarray]:
        if args.task == "wikitext":
            h = hooked(alpha)
            try:
                return {"nll": np.array(nll_each(model, tok, prompts))}
            finally:
                if h:
                    h.remove()
        if args.task == "humaneval":
            return {"pass": np.array([run_humaneval_case("", extract_code(g), r["test"], r["entry_point"])
                                      for g, r in zip(outs, items)], float)}
        return {"distinct3": np.array([distinct3(g) for g in outs]),
                "fluency_nll": np.array(nll_each(model, tok, [p + g for p, g in zip(prompts, outs)])),
                "words": np.array([len(WORD.findall(g.lower())) for g in outs], float)}

    alphas = [float(a) for a in args.alphas.split(",")]
    outs_by = {a: (None if args.task == "wikitext" else run(a)) for a in [None] + alphas}
    if args.task == "wikitext":
        trace = [get_trace_activations(model, tok, "", t)[det["layer"]] for t in prompts]
        prm = [get_activations_all_layers(model, tok, t[:200], pos=pdet["pos"])[pdet["layer"]] for t in prompts]
    else:
        trace = [get_trace_activations(model, tok, p, g)[det["layer"]] for p, g in zip(prompts, outs_by[None])]
        prm = [get_activations_all_layers(model, tok, p, pos=pdet["pos"])[pdet["layer"]] for p in prompts]
    fire_post = np.array([float(x @ det["w"].float() + det["b"]) > tau for x in trace])
    fire_pre = np.array([float(x @ pdet["w"].float() + pdet["b"]) > ptau for x in prm])
    base = per_item(None, outs_by[None])
    arms = {"unsteered": base}
    for a in alphas:
        full = per_item(a, outs_by[a])
        arms[f"shift {a} (ungated)"] = full
        arms[f"PTS post-hoc, shift {a}"] = {k: np.where(fire_post, full[k], base[k]) for k in full}
        arms[f"PTS prompt, shift {a}"] = {k: np.where(fire_pre, full[k], base[k]) for k in full}
    summary = {"task": args.task, "n": len(prompts), "fire_posthoc": float(fire_post.mean()),
               "fire_prompt": float(fire_pre.mean()), "tau": tau, "ptau": ptau, "methods": {}}
    for name, cols in arms.items():
        rec = {}
        for k, x in cols.items():
            d = x - base[k]
            rec[k] = {"mean": round(float(x.mean()), 4), "ci": boot(x), "delta": round(float(d.mean()), 4),
                      "delta_ci": boot(d), "changed": round(float((d != 0).mean()), 4)}
            if k.endswith("nll"):
                rec[k]["ppl"] = round(float(np.exp(x.mean())), 3)
        summary["methods"][name] = rec
        print(name, {k: (r["mean"], r["delta_ci"]) for k, r in rec.items()}, flush=True)
    write_json(RESULTS_DIR / args.outdir / f"{args.task}.json", summary)
    if args.task != "wikitext":
        write_jsonl(RESULTS_DIR / args.outdir / f"{args.task}_outputs.jsonl",
                    [{"i": i, "fire_posthoc": bool(fp), "fire_prompt": bool(fq), **{str(a): outs_by[a][i] for a in outs_by}}
                     for i, (fp, fq) in enumerate(zip(fire_post, fire_pre))])
