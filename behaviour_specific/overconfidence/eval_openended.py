# Capability preservation outside the target behavior: the operators were fitted
# on multiple-choice reasoning, so an edit that is really local should leave
# free-form generation alone. We measure code synthesis (HumanEval pass@1),
# open-ended instruction following (fluency, repetition, length; judged
# separately), and held-out language-model perplexity (WikiText-103).
# Run: python -m behaviour_specific.overconfidence.eval_openended --task humaneval

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_v6_online import load_gate
from general.inference import generate_batch_chunked
from general.online_gate import SequentialGate
from general.paths import DATA_DIR, RESULTS_DIR
from general.steering import (
    bw_map, steer_ablate, steer_add, steer_fullspace_affine, steer_perneuron_affine, steering_hook,
)
from general.storage import write_json, write_jsonl
from models_specific.active import chat_prompt

WORD = __import__("re").compile(r"[a-z']+")


def distinct3(text: str) -> float:
    w = WORD.findall(text.lower())
    if len(w) < 3:
        return 1.0
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(set(tri)) / len(tri)


def run_humaneval_case(prompt: str, completion: str, test: str, entry: str, timeout: int = 8) -> bool:
    """Execute one HumanEval unit test in a fresh interpreter; True iff it passes."""
    prog = f"{prompt}{completion}\n{test}\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def extract_code(text: str) -> str:
    """Body of the first fenced block, else the raw text (models fence by default)."""
    if "```" not in text:
        return text
    part = text.split("```", 2)[1]
    return part.split("\n", 1)[1] if part.startswith("python") else part


@torch.no_grad()
def corpus_nll(model, tok, texts: list[str], max_len: int = 512) -> float:
    """Mean token NLL of `texts` under the (possibly hooked) model."""
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if ids["input_ids"].shape[1] < 2:
            continue
        loss = model(**ids, labels=ids["input_ids"]).loss
        k = ids["input_ids"].shape[1] - 1
        tot += float(loss) * k
        n += k
    return tot / max(n, 1)


def _selftest():
    assert distinct3("a b c a b c") < 1.0 and distinct3("a b c d e f") == 1.0
    assert extract_code("```python\nx=1\n```") .strip() == "x=1"
    assert extract_code("x=2").strip() == "x=2"
    print("eval_openended self-test passed")


def mmlu_norm(model, tok, layer: int) -> float:
    """The ||h|| the additive dose was calibrated against (MMLU prompts, layer L)."""
    from behaviour_specific.overconfidence.benchmarks import load_records
    from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
    from behaviour_specific.overconfidence.steer_overconfidence import mean_activation_norm
    recs = load_records("mmlu", n=300, seed=7)
    return mean_activation_norm(model, tok, recs, layer, lambda r: chat_prompt(tok, mcq_prompt(r)))


def tuned_winner(method: str, parity_dir: str = "v3_parity") -> dict:
    from behaviour_specific.overconfidence.steer_v7_tuned import parse
    p = RESULTS_DIR / parity_dir / f"parity_{method}.json"
    return parse(json.loads(p.read_text())["winner"]) if p.exists() else {}


def build_ops(model, tok, layer, dev, gate_layer=16, decide_at=16, tau_q=0.30):
    pts = torch.load(DIRECTIONS_DIR / f"pts_L{layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{layer}.pt")
    joint = torch.load(DIRECTIONS_DIR / f"joint_stats_L{layer}.pt")
    fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{layer}.pt")
    pn = torch.load(DIRECTIONS_DIR / f"perneuron_stats_L{layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    d = fm["m_s"].shape[0]
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    norm = mmlu_norm(model, tok, layer)
    conds = build_conditions(stats, norm)
    A = bw_map(fm["S_s"].double() + 1e-4 * torch.eye(d, dtype=torch.float64),
               fm["S_t"].double() + 1e-4 * torch.eye(d, dtype=torch.float64)).float().to(dev)
    V_c, w_c, b, tau, sigma = load_gate(gate_layer, tau_q, 0.05, decide_at)
    V, w = V_c.to(dev, torch.float32), w_c.to(dev, torch.float32)
    ops = {
        "baseline": (layer, None),
        "additive": (layer, lambda: (lambda h: steer_add(h, v, -0.75 * norm))),
        "clamp_q50": (layer, lambda: conds["clamp_q50"](v)),
        "quantile-OT": (layer, lambda: conds["otq_cal"](v)),
        "ablation": (layer, lambda: (lambda h: steer_ablate(h, v))),
        "MiMiC": (layer, lambda: (lambda h: steer_fullspace_affine(h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)))),
        "Linear-AcT": (layer, lambda: (lambda h: steer_perneuron_affine(
            h, pn["mu_s"].to(dev), pn["sig_s"].to(dev), pn["mu_t"].to(dev), pn["sig_t"].to(dev), 1.0))),
        "online-gated ablation": (layer, lambda: (SequentialGate(V, w, b, tau, sigma, delta=0.05, warmup=8,
                                                                 decide_at=decide_at),
                                                  lambda h: steer_ablate(h, v))),
    }
    wa = tuned_winner("additive")
    if "alpha" in wa:
        ops["additive (tuned)"] = (layer, lambda a=wa["alpha"]: (lambda h: steer_add(h, v, a * norm)))
    wm = tuned_winner("gatedmimic")
    if "layer" in wm:
        Lm = wm["layer"]
        fmt = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{Lm}.pt")
        At = bw_map(fmt["S_s"].double() + 1e-2 * torch.eye(d, dtype=torch.float64),
                    fmt["S_t"].double() + 1e-2 * torch.eye(d, dtype=torch.float64)).float().to(dev)
        mimic_t = lambda h: steer_fullspace_affine(h, fmt["m_s"].to(dev), At, fmt["m_t"].to(dev))
        ops["MiMiC (tuned)"] = (Lm, lambda: mimic_t)
        ops["gated full-rank transport"] = (Lm, lambda: (
            SequentialGate(V, w, b, tau, sigma, delta=0.05, warmup=8, decide_at=decide_at), mimic_t))
    return ops


class hooked:
    """`with hooked(model, layer, spec, gate_layer)`: plain op, or gate reader + actor.

    `spec` is None (unsteered), an `fn(h)->h'`, or a `(SequentialGate, action)` pair.
    """

    def __init__(self, model, layer, spec, gate_layer):
        self.model, self.layer, self.spec, self.gate_layer = model, layer, spec, gate_layer
        self.handles = []

    def __enter__(self):
        if self.spec is None:
            return self
        if isinstance(self.spec, tuple):
            gate, action = self.spec
            self.handles = [steering_hook(self.model.model.layers[self.layer], gate.actor(action)),
                            steering_hook(self.model.model.layers[self.gate_layer], gate.reader)]
        else:
            self.handles = [steering_hook(self.model.model.layers[self.layer], self.spec)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def fire_stats(spec) -> dict:
    """Fire rate of a sequential gate (empty for unconditional operators)."""
    if not isinstance(spec, tuple):
        return {}
    steps = [x for x in spec[0].fire_steps() if x is not None]
    n = max(len(spec[0].fire_steps()), 1)
    return {"fire_rate": round(len(steps) / n, 4),
            "fire_step_median": int(sorted(steps)[len(steps) // 2]) if steps else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="humaneval", help="humaneval | openended | wikitext")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--gate-layer", type=int, default=16)
    ap.add_argument("--decide-at", type=int, default=16)
    ap.add_argument("--tau-q", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=0, help="0 = whole task set")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--outdir", default="v3_openended")
    args = ap.parse_args()

    _selftest()

    from datasets import load_dataset
    from models_specific.active import load_model

    model, tok = load_model()
    dev = model.device
    ops = build_ops(model, tok, args.layer, dev, args.gate_layer, args.decide_at, args.tau_q)
    out_dir = RESULTS_DIR / args.outdir
    summary = {"task": args.task, "layer": args.layer, "methods": {}}

    if args.task == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t) > 400][: (args.n or 300)]
        for name, (op_layer, make) in ops.items():
            spec = make() if make else None
            with hooked(model, op_layer, spec, args.gate_layer):
                nll = corpus_nll(model, tok, texts)
            summary["methods"][name] = {"nll": round(nll, 4),
                                        "ppl": round(float(torch.tensor(nll).exp()), 3),
                                        **fire_stats(spec)}
            print(f"  {name:24s} ppl {summary['methods'][name]['ppl']}", flush=True)

    elif args.task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = list(ds)[: (args.n or len(ds))]
        instr = ("Complete the following Python function. Reply with the full function "
                 "in a single ```python code block and nothing else.\n\n")
        prompts = [chat_prompt(tok, instr + r["prompt"]) for r in rows]
        for name, (op_layer, make) in ops.items():
            spec = make() if make else None
            with hooked(model, op_layer, spec, args.gate_layer):
                gens = generate_batch_chunked(model, tok, prompts, batch_size=args.batch_size,
                                              max_new_tokens=args.max_new_tokens)
            ok = [run_humaneval_case("", extract_code(g), r["test"], r["entry_point"])
                  for g, r in zip(gens, rows)]
            rec = {"pass@1": round(sum(ok) / len(ok), 4), "n": len(ok),
                   "mean_chars": round(sum(len(g) for g in gens) / len(gens), 1),
                   **fire_stats(spec)}
            summary["methods"][name] = rec
            write_jsonl(out_dir / f"humaneval_{name.replace(' ', '_')}.jsonl",
                        [{"task_id": r["task_id"], "passed": p, "completion": g}
                         for r, p, g in zip(rows, ok, gens)])
            print(f"  {name:24s} {rec}", flush=True)

    elif args.task == "openended":
        src = DATA_DIR / "openended_prompts.json"
        instructions = json.loads(src.read_text())[: (args.n or 200)]
        prompts = [chat_prompt(tok, t) for t in instructions]
        base_gens = None
        for name, (op_layer, make) in ops.items():
            spec = make() if make else None
            with hooked(model, op_layer, spec, args.gate_layer):
                gens = generate_batch_chunked(model, tok, prompts, batch_size=args.batch_size,
                                              max_new_tokens=args.max_new_tokens)
            if base_gens is None:
                base_gens = gens
            nll = corpus_nll(model, tok, [p + g for p, g in zip(prompts, gens)])
            d3 = [distinct3(g) for g in gens]
            rec = {"fluency_ppl": round(float(torch.tensor(nll).exp()), 3),
                   "distinct3": round(sum(d3) / len(d3), 4),
                   "pct_degenerate": round(sum(x < 0.5 for x in d3) / len(d3), 4),
                   "mean_words": round(sum(len(WORD.findall(g.lower())) for g in gens) / len(gens), 1),
                   "n": len(gens), **fire_stats(spec)}
            summary["methods"][name] = rec
            write_jsonl(out_dir / f"openended_{name.replace(' ', '_')}.jsonl",
                        [{"instruction": t, "output": g} for t, g in zip(instructions, gens)])
            print(f"  {name:24s} {rec}", flush=True)
    else:
        raise SystemExit(f"unknown task {args.task}")

    write_json(out_dir / f"{args.task}.json", summary)
    print("done ->", out_dir / f"{args.task}.json")
