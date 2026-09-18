import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/materialize_irc_data.py"
    assert path.is_file()
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        return runpy.run_path(str(path))


def test_exposure_reader_uses_only_declared_text_fields(tmp_path):
    code = module()
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"rows": [{"prompt": "Problem A", "completion": "Gold secret"}]}))
    spec = {"path": str(path), "format": "nested_json", "fields": ["prompt"]}
    assert code["read_exposure"](spec, None) == ["Problem A"]


def test_group_partition_excludes_history_and_train_test_overlap():
    code = module()
    rows = [
        {"row_id": str(i), "split": split, "reference_eligible": True}
        for i, split in enumerate(["train", "train", "test", "test", "test", "train"])
    ]
    groups = ["train-a", "shared", "shared", "fresh-test", "exposed", "train-a"]
    result = code["partition_math"](rows, groups, {"exposed"}, {"fit": 2})
    assert [row["allocation"] for row in result] == [
        "fit",
        "fit",
        "excluded_training_overlap",
        "sealed_test",
        "excluded_history",
        "duplicate",
    ]
    assert len({row["cluster_id"] for row in result if row["allocation"] == "fit"}) == 2


def test_group_partition_rejects_insufficient_quota_and_preserves_invalid_reference():
    code = module()
    rows = [{"row_id": "a", "split": "train", "reference_eligible": False}]
    with pytest.raises(ValueError, match="insufficient"):
        code["partition_math"](rows, ["group"], set(), {"fit": 1})
    result = code["partition_math"](rows, ["group"], set(), {})
    assert result[0]["allocation"] == "excluded_reference"


def test_materializer_rejects_existing_output_before_loading_sources(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "payload"
    protected.write_text("preserve")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_irc_data.py",
            "--exposure-spec",
            "missing.json",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert protected.read_text() == "preserve"
