from __future__ import annotations


from behaviour_specific.overconfidence.mmlu.data import load_mmlu

OCW = "overconfident_wrong"
CR = "confident_right"
EVAL_SEEDS = (7, 11, 23, 31, 47)


def tuning_records(n: int, seed: int, eval_n: int = 300, extract_n: int = 400,
                   extract_seed: int = 2) -> list[dict]:
    used = {r["id"] for s in EVAL_SEEDS for r in load_mmlu(n=eval_n, seed=s)}
    used |= {r["id"] for r in load_mmlu(n=extract_n, seed=extract_seed)}
    return [r for r in load_mmlu(seed=seed) if r["id"] not in used][:n]


def surgical(before: list[dict], after: list[dict]) -> dict:
    a = {r["id"]: r for r in after}
    ocw = [r for r in before if r["state"] == OCW]
    cr = [r for r in before if r["state"] == CR]
    rm = sum(a[r["id"]]["state"] != OCW for r in ocw) / max(len(ocw), 1)
    keep = sum(a[r["id"]]["state"] == CR for r in cr) / max(len(cr), 1)
    acc_b = sum(r["is_correct"] for r in before) / len(before)
    acc_a = sum(a[r["id"]]["is_correct"] for r in before) / len(before)
    return {"ocw_rm": round(rm, 4), "cr_keep": round(keep, 4),
            "selectivity": round(rm - (1 - keep), 4), "d_acc": round(acc_a - acc_b, 4),
            "n_ocw": len(ocw), "n_cr": len(cr)}


def pick(results: dict, acc_floor: float = -0.01) -> str:
    ok = [k for k, v in results.items() if v["d_acc"] >= acc_floor] or list(results)
    return max(ok, key=lambda k: (results[k]["selectivity"], results[k]["d_acc"]))


def _selftest():
    b = [{"id": 1, "state": OCW, "is_correct": False}, {"id": 2, "state": CR, "is_correct": True}]
    a = [{"id": 1, "state": CR, "is_correct": True}, {"id": 2, "state": CR, "is_correct": True}]
    s = surgical(b, a)
    assert s["ocw_rm"] == 1.0 and s["cr_keep"] == 1.0 and s["selectivity"] == 1.0
    assert pick({"x": {"selectivity": .1, "d_acc": 0.0}, "y": {"selectivity": .3, "d_acc": -0.5}}) == "x"
    print("sweep_parity self-test passed")
