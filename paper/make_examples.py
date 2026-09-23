from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "v4_pooled" / "examples_gemma.json"
OUT = ROOT / "paper" / "gen" / "examples.tex"
TITLES = {"removed": "overconfident-wrong, state changed", "missed": "overconfident-wrong, state unchanged",
          "kept": "confident-right, kept", "lost": "confident-right, lost"}
STATE = {"overconfident_wrong": "OCW", "nonconfident_wrong": "NCW", "nonconfident_right": "NCR", "confident_right": "CR"}


def tex(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().encode("ascii", "replace").decode()
    for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}")):
        s = s.replace(a, b)
    return s


def clip(s: str, head: int = 420, tail: int = 160) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= head + tail + 20 else s[:head] + " [...] " + s[-tail:]


def side(label: str, v: dict) -> str:
    return (f"\\textbf{{{label}}} (answer {v['answer']}, $P(\\mathrm{{YES}})={v['p_yes']:.2f}$, {STATE[v['state']]}): "
            f"\\texttt{{{tex(clip(v['trace']))}}}")


if __name__ == "__main__":
    d = json.loads(SRC.read_text())
    parts = []
    for cell, rows in d["cells"].items():
        for r in rows:
            opts = " ".join(f"{L}. {tex(o)}" for L, o in zip("ABCD", r["options"]))
            parts.append(f"\\noindent\\textbf{{{TITLES[cell]}}} (\\texttt{{{r['id']}}}, gold {r['gold']}).\n"
                         f"\\begin{{quote}}\\scriptsize\\emph{{Question:}} {tex(r['question'])} {opts}\\\\[2pt]\n"
                         f"{side('before', r['before'])}\\\\[2pt]\n{side('after', r['after'])}\n\\end{{quote}}\n")
    OUT.write_text("\n".join(parts))
    print(d["counts"], len(parts))
