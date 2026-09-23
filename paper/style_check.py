from __future__ import annotations

import re
from pathlib import Path

TARGET = {"semicolons": (0.6, 3.4), "colons": (0.0, 12.0), "parentheses": (7.5, 20.0),
          "em-dashes": (0.0, 2.5), "we": (15.0, 32.0)}


MACROS = {r"\ocw{}": "overconfident-wrong", r"\crok{}": "confident-right"}


def prose(tex: str) -> str:
    tex = tex[tex.index(r"\begin{abstract}"):tex.index(r"\begin{thebibliography}")]
    for k, v in MACROS.items():
        tex = tex.replace(k, v)
    for env in ("figure", r"figure\*", "table", r"table\*", "tabular", "tikzpicture", "axis"):
        tex = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), " ", tex, flags=re.S)
    tex = re.sub(r"(?m)(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\looseness\s*=\s*-?\d+", " ", tex)
    tex = re.sub(r"(?m)^\\textbf\{[^{}]*\}", " ", tex)
    tex = re.sub(r"\\textbf\{(Mark|Alexander|Anna|Ekaterina|Maria)[^{}]*\}[^.]*\.", " ", tex)
    for _ in range(3):
        tex = re.sub(r"\\(emph|textbf|textit|texttt|mbox)\{([^{}]*)\}", r"\2", tex)
    def _num(m):
        inner = m.group(1).replace(r"\times", "x").replace(r"\%", "%").replace("{", "").replace("}", "")
        inner = inner.replace(r"\pm", "+-").replace("\\,", "").strip()
        return f" {inner} " if re.fullmatch(r"[-+0-9.,%x\s]+", inner) else " X "
    tex = re.sub(r"\$([^$]*)\$", _num, tex)
    tex = re.sub(r"\\begin\{(remark|theorem|corollary|proposition|definition|lemma)\}\[[^\]]*\]",
                 " ", tex)
    tex = tex.replace("``", '"').replace("''", '"')
    tex = re.sub(r"(Sec|Table|Appendix|Fig|Thm|Prop|Cor|Eq)\w*\.?~?\\ref\{[^}]*\}",
                 r"\1 1", tex)
    tex = re.sub(r"\\(cite|ref|eqref|label|verb)\w*\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", tex)
    tex = re.sub(r"[{}~]", " ", tex)
    tex = re.sub(r"\(\s*\)", " ", tex)
    tex = re.sub(r"\s+([,.;:])", r"\1", tex)
    return re.sub(r"\s+", " ", tex)


def report(path: Path) -> None:
    s = prose(path.read_text())
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", s.lower())
    n = len(words)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", s) if len(x.split()) >= 4]
    wl = [len(x.split()) for x in sents]
    counts = {"semicolons": s.count(";"), "colons": s.count(":"), "parentheses": s.count("("),
              "em-dashes": s.count("---"), "we": len(re.findall(r"\bwe\b", s, re.I))}
    print(f"{n} words, {len(sents)} sentences, mean {sum(wl) / len(wl):.1f}, "
          f"median {sorted(wl)[len(wl) // 2]}, >45w {100 * sum(x > 45 for x in wl) / len(wl):.1f}%")
    for k, c in counts.items():
        rate = 1000 * c / n
        lo, hi = TARGET[k]
        flag = "OK " if lo <= rate <= hi else ("LOW" if rate < lo else "HIGH")
        print(f"  {k:12s} {c:4d}  {rate:6.1f}/1k  target {lo}-{hi}  {flag}")
    banned = ["furthermore", "crucial", "delve", "as expected", "it is worth noting",
              "leverage", "note that", "clearly", "obviously", "significantly"]
    hits = {b: len(re.findall(rf"\b{b}\b", s, re.I)) for b in banned}
    print("  banned:", {k: v for k, v in hits.items() if v})


if __name__ == "__main__":
    report(Path(__file__).resolve().parent / "paper.tex")
