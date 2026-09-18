# Streaming JSONL + JSON I/O for saving rollouts and run summaries.
#
# The big runs (full MMLU) produce hundreds of thousands of rich rollout records;
# `JsonlWriter` streams them to disk one at a time (flushing as it goes) so a run
# that is interrupted still leaves everything computed so far, and memory stays
# flat. Every method saves, per question, the system/user prompts + full
# generations + parsed answer, so results can be checked by hand.
#
# `python -m general.storage` runs the self-tests (no model).

from __future__ import annotations

import json
from pathlib import Path


class JsonlWriter:
    """Append-as-you-go jsonl writer. Use as a context manager."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None

    def __enter__(self):
        self._f = self.path.open("w")
        return self

    def write(self, row: dict) -> None:
        """Write one record and flush (so partial runs are recoverable)."""
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def __exit__(self, *exc):
        if self._f:
            self._f.close()


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts as jsonl (parents created)."""
    with JsonlWriter(path) as w:
        for r in rows:
            w.write(r)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a jsonl file into a list of dicts.

    Splits on "\n" ONLY: rows are written with ensure_ascii=False, so free text
    may contain U+2028/U+0085-style separators that str.splitlines() would
    (wrongly) treat as record boundaries.
    """
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def write_json(path: str | Path, obj: dict) -> None:
    """Write a single JSON object, pretty-printed (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: str | Path) -> dict:
    """Read a single JSON object."""
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "x.jsonl"
        with JsonlWriter(p) as w:
            w.write({"a": 1})
            w.write({"a": 2})
        assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
        write_jsonl(p, [{"b": 3}])
        assert read_jsonl(p) == [{"b": 3}]
        write_json(Path(d) / "s.json", {"ok": True})
        assert read_json(Path(d) / "s.json") == {"ok": True}
    print("general.storage self-tests passed")
