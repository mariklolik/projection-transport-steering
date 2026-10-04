from __future__ import annotations

import json
from pathlib import Path


class JsonlWriter:


    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None

    def __enter__(self):
        self._f = self.path.open("w")
        return self

    def write(self, row: dict) -> None:

        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def __exit__(self, *exc):
        if self._f:
            self._f.close()


def write_jsonl(path: str | Path, rows: list[dict]) -> None:

    with JsonlWriter(path) as w:
        for r in rows:
            w.write(r)


def read_jsonl(path: str | Path) -> list[dict]:

    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def write_json(path: str | Path, obj: dict) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: str | Path) -> dict:

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
