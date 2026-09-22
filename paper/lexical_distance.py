# Vocabulary distance between the manuscript's prose and the reference corpus:
# Jensen-Shannon divergence over unigrams, type-token ratio, and the content
# words that are most over- and under-used relative to the references.
#   python3 paper/lexical_distance.py <ref.md> [<ref.md> ...]

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

from style_check import prose

STOP = set("""a an the of to in and or is are was were be been being for on at by with as that this
those these it its from not no but if then than so such we our us they their he she his her you your
i me my which who whom whose what when where how why can could may might must shall should will would
do does did done have has had having there here also more most much many any all each other some""".split())


ABBREV = re.compile(r"\b(et al|Fig|Sec|Eq|Thm|Prop|Cor|cf|e\.g|i\.e|vs|Dr|approx)\.$")


def sentences(text: str) -> list[str]:
    """Split on terminal punctuation, but not on the common abbreviations."""
    out, buf = [], ""
    for part in re.split(r"(?<=[.!?])\s+", text):
        buf = (buf + " " + part).strip() if buf else part
        if not ABBREV.search(buf):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]+", text.lower())


def jsd(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    np_, nq = sum(p.values()), sum(q.values())
    out = 0.0
    for k in keys:
        a, b = p[k] / np_, q[k] / nq
        m = 0.5 * (a + b)
        if a:
            out += 0.5 * a * math.log2(a / m)
        if b:
            out += 0.5 * b * math.log2(b / m)
    return out


def md_prose(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+|\$[^$]*\$|\{[^{}]*\}", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text)


def _selftest():
    a, b = Counter("aabb"), Counter("aabb")
    assert jsd(a, b) < 1e-9
    assert jsd(Counter("aaaa"), Counter("bbbb")) > 0.99
    print("lexical_distance self-test passed")


if __name__ == "__main__":
    _selftest()
    root = Path(__file__).resolve().parent
    ours = Counter(w for w in words(prose((root / "paper.tex").read_text())) if w not in STOP)
    refs = {}
    for p in sys.argv[1:]:
        refs[Path(p).stem] = Counter(
            w for w in words(md_prose(Path(p).read_text())) if w not in STOP)
    pooled = Counter()
    for c in refs.values():
        pooled.update(c)

    print(f"ours: {sum(ours.values())} content tokens, {len(ours)} types, "
          f"TTR {len(ours) / sum(ours.values()):.3f}")
    for name, c in refs.items():
        print(f"  {name}: JSD(ours,ref) {jsd(ours, c):.4f}  TTR {len(c) / sum(c.values()):.3f}")
    print(f"  pooled reference corpus: JSD {jsd(ours, pooled):.4f}")
    pair = [jsd(a, b) for i, a in enumerate(refs.values()) for b in list(refs.values())[i + 1:]]
    print(f"  reference-to-reference JSD: min {min(pair):.4f} mean "
          f"{sum(pair) / len(pair):.4f} max {max(pair):.4f}")

    # Content words carry the topic, so the distance above is mostly subject
    # matter. Function words are the topic-independent stylometric signal.
    fours = Counter(w for w in words(prose((root / "paper.tex").read_text())) if w in STOP)
    frefs = {k: Counter(w for w in words(md_prose(Path(v).read_text())) if w in STOP)
             for k, v in zip(refs, sys.argv[1:])}
    fpool = Counter()
    for c in frefs.values():
        fpool.update(c)
    print(f"\nfunction-word (stylometric) distance, {sum(fours.values())} tokens:")
    for name, c in frefs.items():
        print(f"  {name}: JSD {jsd(fours, c):.4f}")
    print(f"  pooled reference corpus: JSD {jsd(fours, fpool):.4f}")
    fpair = [jsd(a, b) for i, a in enumerate(frefs.values()) for b in list(frefs.values())[i + 1:]]
    print(f"  reference-to-reference JSD: min {min(fpair):.4f} mean "
          f"{sum(fpair) / len(fpair):.4f} max {max(fpair):.4f}")
    nf, npf = sum(fours.values()), sum(fpool.values())
    dev = sorted(((w, 1e4 * fours[w] / nf, 1e4 * fpool[w] / npf) for w in fpool if fpool[w] >= 60),
                 key=lambda x: -abs(x[1] - x[2]))[:10]
    print("  largest function-word gaps (ours/corpus per 10k):",
          ", ".join(f"{w} {a:.0f}/{b:.0f}" for w, a, b in dev))

    no, npool = sum(ours.values()), sum(pooled.values())
    rate = [(w, 1e4 * c / no, 1e4 * pooled[w] / npool) for w, c in ours.items() if c >= 8]
    over = sorted(rate, key=lambda x: -(x[1] - x[2]))[:12]
    print("  most over-used vs corpus (per 10k):",
          ", ".join(f"{w} {a:.0f}/{b:.0f}" for w, a, b in over))
    common = sorted((w for w, c in pooled.items() if c >= 40),
                    key=lambda w: -(1e4 * pooled[w] / npool - 1e4 * ours[w] / no))[:12]
    print("  corpus words we under-use:",
          ", ".join(f"{w} {1e4 * ours[w] / no:.0f}/{1e4 * pooled[w] / npool:.0f}" for w in common))
