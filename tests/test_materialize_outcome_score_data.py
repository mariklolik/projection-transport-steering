import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_materializer_writes_stage_separated_hash_bound_packets(tmp_path):
    rows = [
        {
            "year": 2000 + index // 50,
            "index": index,
            "part": "I",
            "problem": f"p{index}",
            "answer": str(index % 1000),
        }
        for index in range(975)
    ]
    rows.extend(
        {
            "year": year,
            "index": index,
            "part": "I",
            "problem": f"p{year}-{index}",
            "answer": str(index),
        }
        for year in (2024, 2025)
        for index in range(30)
    )
    source = tmp_path / "aime.parquet"
    output = tmp_path / "allocation"
    pq.write_table(pa.Table.from_pylist(rows), source)

    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_outcome_score_data.py",
            "--historical",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )

    counts = {
        path.stem: len(json.loads(path.read_text())["rows"]) for path in output.glob("*.json")
    }
    assert counts == {
        "basis": 32,
        "calibration": 128,
        "development": 30,
        "fit": 256,
        "pilot": 30,
        "reserve": 431,
        "validation": 128,
    }
    assert all(json.loads(path.read_text())["source_sha256"] for path in output.glob("*.json"))
