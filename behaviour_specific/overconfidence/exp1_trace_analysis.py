from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean, median

from general.paths import RESULTS_DIR
from general.storage import read_jsonl, write_json

HEDGE = {"maybe", "perhaps", "might", "possibly", "however", "wait", "actually", "hmm",
         "unsure", "unclear", "guess", "seems", "probably", "could", "uncertain", "confused"}
CERTAINTY = {"clearly", "obviously", "definitely", "certainly", "surely", "undoubtedly",
             "evidently", "must", "certain", "sure"}
WORD = re.compile(r"[a-z']+")


def distinct3(words: list[str]) -> float:
    if len(words) < 3:
        return 1.0
    tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(tri)) / len(tri)


def trace_metrics(traces: list[str], token_lens: list[int] | None = None) -> dict:
    n = len(traces)
    words = [WORD.findall(t.lower()) for t in traces]
    wlen = [len(w) for w in words]
    d3 = [distinct3(w) for w in words]
    total_w = sum(wlen) or 1
    hedge = sum(sum(x in HEDGE for x in w) for w in words)
    cert = sum(sum(x in CERTAINTY for x in w) for w in words)
    out = {
        "n": n,
        "mean_chars": round(mean(len(t) for t in traces), 1),
        "median_chars": int(median(len(t) for t in traces)),
        "mean_words": round(mean(wlen), 1),
        "mean_distinct3": round(mean(d3), 3),
        "pct_degenerate": round(sum(x < 0.5 for x in d3) / n, 3),
        "close_rate": round(sum("</think>" in t for t in traces) / n, 3),
        "boxed_rate": round(sum("\\boxed{" in t for t in traces) / n, 3),
        "hedge_per_100w": round(100 * hedge / total_w, 3),
        "hedge_per_trace": round(hedge / n, 3),
        "certainty_per_trace": round(cert / n, 3),
        "certainty_per_100w": round(100 * cert / total_w, 3),
        "pct_any_hedge": round(sum(any(x in HEDGE for x in w) for w in words) / n, 3),
    }
    if token_lens:
        out["mean_tokens"] = round(mean(token_lens), 1)
    return out


def condition_traces(rows: list[dict]) -> tuple[list[str], list[int] | None]:
    traces = [r["generations"][0]["text"] for r in rows]
    toks = [r["reasoning_tokens"] for r in rows if "reasoning_tokens" in r]
    return traces, (toks if len(toks) == len(rows) else None)


def parse_alpha(tag: str) -> str:
    if tag.startswith("baseline"):
        return "0"
    m = re.search(r"_a([+-]\d+\.\d+)_", tag)
    if m:
        return m.group(1)
    if "_think_" in tag:
        return "think"
    if "_answer_" in tag:
        return "answer"
    if "ablate" in tag:
        return "ablate"
    return "?"


def _selftest():
    m = trace_metrics(["<think>maybe it is A, wait no</think> \\boxed{A}", "a a a a a a"])
    assert m["n"] == 2 and 0 <= m["mean_distinct3"] <= 1
    assert m["close_rate"] == 0.5 and m["boxed_rate"] == 0.5
    assert m["hedge_per_100w"] > 0
    assert distinct3(["a"] * 8) < 0.5
    assert parse_alpha("exp1_caa_m1_a-1.00_m2") == "-1.00" and parse_alpha("baseline_m2") == "0"
    print("exp1_trace_analysis self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-dir", default="steering_exp1")
    args = ap.parse_args()

    _selftest()

    roll = RESULTS_DIR / args.steer_dir / "rollouts"
    files = sorted(roll.glob("*_m2__shard*.jsonl"))
    if not files:
        raise SystemExit(f"no *_m2 rollouts under {roll}")
    pooled: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        pooled[f.stem.rsplit("__shard", 1)[0]].extend(read_jsonl(f))

    metrics = {}
    for tag, rows in pooled.items():
        traces, toks = condition_traces(rows)
        metrics[tag] = {"alpha": parse_alpha(tag), **trace_metrics(traces, toks)}

    def vec_of(tag: str) -> str:
        return "baseline" if tag.startswith("baseline") else tag.rsplit("_", 1)[0].rsplit("_a", 1)[0] \
            .replace("_ablate", "").replace("_think", "").replace("_answer", "")

    order = {"0": 0, "-2.00": -20, "-1.50": -15, "-1.00": -10, "-0.50": -5,
             "+0.50": 5, "+1.00": 10, "+1.50": 15, "+2.00": 20, "ablate": 100, "think": 101, "answer": 102}
    base = metrics.get("baseline_m2", {})
    print(f"baseline_m2: {base.get('mean_chars')} chars / {base.get('mean_words')} words, "
          f"close {base.get('close_rate')}, boxed {base.get('boxed_rate')}, "
          f"hedge/100w {base.get('hedge_per_100w')}, distinct3 {base.get('mean_distinct3')}")
    print(f"\n{'condition':34s} {'α':>6} {'words':>6} {'close':>6} {'boxed':>6} "
          f"{'dist3':>6} {'degen%':>7} {'hedge':>6} {'certain':>7}")
    for tag in sorted(metrics, key=lambda t: (vec_of(t), order.get(metrics[t]["alpha"], 0))):
        m = metrics[tag]
        print(f"{tag:34s} {m['alpha']:>6} {m['mean_words']:>6.0f} {m['close_rate']:>6.2f} "
              f"{m['boxed_rate']:>6.2f} {m['mean_distinct3']:>6.2f} {m['pct_degenerate']:>7.2f} "
              f"{m['hedge_per_100w']:>6.2f} {m['certainty_per_100w']:>7.2f}")

    SHOWCASE = ["baseline_m2", "exp1_caa_m3_a+0.50_m2", "exp1_caa_m1_a+0.50_m2",
                "exp1_caa_m3_a-0.50_m2", "exp1_caa_m4_a-0.50_m2", "exp1_caa_m1_a-1.00_m2"]
    CLIP = 900

    def hc(trace: str) -> tuple[int, int]:
        w = WORD.findall(trace.lower())
        return sum(x in HEDGE for x in w), sum(x in CERTAINTY for x in w)

    same_q = {}
    if pooled.get("baseline_m2"):
        for qi in range(min(3, len(pooled["baseline_m2"]))):
            qid = pooled["baseline_m2"][qi]["id"]
            row = {}
            for tag in SHOWCASE:
                r = next((x for x in pooled.get(tag, []) if x["id"] == qid), None)
                if r:
                    row[tag] = r["generations"][0]["text"][:CLIP]
            same_q[qid] = row

    exemplars = {}
    for tag in SHOWCASE:
        rows = pooled.get(tag, [])
        if not rows:
            continue
        scored = [(hc(x["generations"][0]["text"]), x) for x in rows]
        th = max(scored, key=lambda s: s[0][0])
        tc = max(scored, key=lambda s: s[0][1])
        exemplars[tag] = {
            "top_hedge": {"id": th[1]["id"], "n_hedge": th[0][0], "trace": th[1]["generations"][0]["text"][:CLIP]},
            "top_certainty": {"id": tc[1]["id"], "n_certainty": tc[0][1],
                              "trace": tc[1]["generations"][0]["text"][:CLIP]},
        }

    write_json(RESULTS_DIR / args.steer_dir / "trace_analysis.json",
               {"steer_dir": args.steer_dir, "metrics": metrics,
                "examples_same_question": same_q, "exemplars": exemplars})
    print(f"\nsaved {RESULTS_DIR / args.steer_dir / 'trace_analysis.json'}")
