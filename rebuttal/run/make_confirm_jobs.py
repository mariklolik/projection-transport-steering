from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--parity-dir", required=True)
    ap.add_argument("--env", default="")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--nshards", type=int, default=12)
    ap.add_argument("--split", default="confirm")
    a = ap.parse_args()

    w = json.loads(Path(a.selection).read_text())["winners"]
    arms = [("--methods prompt", "prompt_hedge", "prompting", "1", True, "m5,m4"),
            ("--methods additive", "tuned_additive", "additive CAA \\citep{rimsky2024caa}", "8", True, "m5,m4"),
            ("--methods plain_ablate", "plain_ablate", "directional ablation \\citep{arditi2024refusal}", "1", True, "m5,m4"),
            ("--methods mimic", "tuned_mimic", "MiMiC \\citep{singh2024mimic}", "9", True, "m5,m4"),
            ("--methods act", "tuned_act", "Linear-AcT \\citep{rodriguez2025act}", "9", True, "m5,m4"),
            (f"--methods none --cast-configs {w['castdim']}", f"castdim_{w['castdim']}", "CAST, prompt condition \\citep{lee2024cast}", "16", True, "m5,m4"),
            ("--methods cast", "tuned_cast", "CAST-style, trace condition", "16", True, "m5,m4"),
            (f"--methods none --detector-configs {w['pts']}", f"det_{w['pts']}", "PTS, post-hoc (two passes)", "16", False, "m5,m4"),
            (f"--methods none --detector-configs {w['online']}", f"detonline_{w['online']}", "PTS, decides at a prefix (one pass)", "16", False, "m5"),
            (f"--methods none --prompt-configs {w['castprompt']}", f"detprompt_{w['castprompt']}", "PTS, decides at the prompt (one pass)", "16", False, "m5,m4")]
    base = (f"{a.env} .venv/bin/python -m behaviour_specific.overconfidence.steer_v7_tuned --split {a.split} "
            f"--nshards {a.nshards} --outdir {a.outdir} --parity-dir {a.parity_dir} --detector detector_m5")
    jobs = [f"{base} --shard {k} {flags} --readouts {ro}" for flags, _, _, _, _, ro in arms for k in range(a.nshards)]
    Path(a.jobs).write_text("\n".join(jobs) + "\n")
    Path(a.protocol).write_text(json.dumps({
        "ref": f"det_{w['pts']}", "readout": "m5", "split": a.split,
        "arms": [{"tag": t, "name": n, "budget": b, "primary": p} for _, t, n, b, p, _ in arms]}, indent=1))
    print(len(jobs), "jobs; ref", f"det_{w['pts']}")
