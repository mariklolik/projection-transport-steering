#!/usr/bin/env python3
"""Fan a file of shell job-chains out over (node, gpu) slots and launch them detached.

    python3 rebuttal/run/dispatch.py rebuttal/run/stage2.jobs
    python3 rebuttal/run/dispatch.py --status
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REMOTE = "~/pts-rebuttal"
SLOTS = [(h, g) for h in ("avi-gn-fsk39", "avi-gn-fsk40", "avi-gn-fsk41", "avi-gn-fsk43") for g in range(8)]
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25"]
MIN_FREE_MIB = 30000


def free_slots() -> list[tuple[str, int]]:
    out = []
    for host in sorted({h for h, _ in SLOTS}):
        r = subprocess.run(SSH + [host, "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"],
                           capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            idx, mem = line.split(",")
            if int(mem.strip().split()[0]) > MIN_FREE_MIB:
                out.append((host, int(idx)))
    return out


def launch(jobfile: Path, wait: bool = False) -> None:
    jobs = [l for l in jobfile.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if wait:
        return drain(jobfile, jobs)
    slots = free_slots()
    if len(slots) < len(jobs):
        sys.exit(f"{len(jobs)} jobs but only {len(slots)} free GPUs")
    for i, (job, (host, gpu)) in enumerate(zip(jobs, slots)):
        send(jobfile, i, job, host, gpu)


def drain(jobfile: Path, jobs: list[str], poll: int = 120) -> None:
    """Dispatch the queue as GPUs free up; blocks until every job is launched."""
    import time
    pending = list(enumerate(jobs))
    while pending:
        slots = free_slots()
        for (i, job), (host, gpu) in zip(list(pending), slots):
            send(jobfile, i, job, host, gpu)
            pending.pop(0)
        if pending:
            print(f"  {len(pending)} queued, waiting for a GPU", flush=True)
            time.sleep(poll)


def send(jobfile: Path, i: int, job: str, host: str, gpu: int) -> None:
    tag = f"{jobfile.stem}_{i:02d}"
    script = (f"cd {REMOTE} && export PYTHONPATH=. HF_HUB_OFFLINE=1 "
              f"CUDA_VISIBLE_DEVICES={gpu} TOKENIZERS_PARALLELISM=false "
              f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && "
              f"echo START $(date -u +%H:%M:%S) && {job} && echo JOB_DONE")
    # tmux, not nohup: nodes held in a SLURM reservation reap detached processes
    # within minutes, while tmux sessions survive there.
    inner = f"bash -c {shq(script)} > logs/{tag}.log 2>&1"
    remote = f"cd {REMOTE} && tmux new-session -d -s {tag} {shq(inner)}"
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
                        host, remote], capture_output=True, text=True, timeout=60)
    if r.returncode:
        print(f"{tag:22s} skipped ({r.stderr.strip()[:60]})", flush=True)
        return
    print(f"{tag:22s} {host}:{gpu}  {job[:90]}", flush=True)


def shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def status() -> None:
    for host in sorted({h for h, _ in SLOTS}):
        r = subprocess.run(SSH + [host, f"cd {REMOTE}/logs 2>/dev/null && "
                                        "for f in *.log; do printf '%-26s %s\\n' \"$f\" "
                                        "\"$(grep -c JOB_DONE $f) $(tail -1 $f | cut -c1-70)\"; done"],
                           capture_output=True, text=True)
        print(f"===== {host} =====\n{r.stdout}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile", nargs="?")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--wait", action="store_true", help="queue jobs until GPUs free up")
    a = ap.parse_args()
    status() if a.status or not a.jobfile else launch(Path(a.jobfile), a.wait)
