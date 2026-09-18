import gc
import importlib.util
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_diagnostic(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_attention_diagnostic.py"
    assert path.exists(), "external diagnostic runner is not implemented"
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("attention_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper(fail_at=None):
    query, key, value = [torch.ones(2, 1, 3, 4).requires_grad_() for _ in range(3)]
    q64, k64, v64 = [x.detach().double().requires_grad_() for x in (query, key, value)]
    mask = torch.ones(2, 3, dtype=torch.bool)
    keep = torch.ones(2, 3, 3, dtype=torch.bool).tril()
    valid = keep.any(-1)
    block = SimpleNamespace(BLOCK_SIZE=(128, 128))
    for prefix in ("kv", "q", "full_kv", "full_q"):
        setattr(block, prefix + "_num_blocks", torch.ones(2, 1, 1, dtype=torch.int32))
        setattr(block, prefix + "_indices", torch.zeros(2, 1, 1, 1, dtype=torch.int32))
    output = query + key + value
    oracle = q64 + k64 + v64
    torch.testing.assert_close(output.double(), oracle, rtol=0.005, atol=0.0005)
    inference_output = output.detach()
    torch.testing.assert_close(inference_output.double(), oracle, rtol=0.005, atol=0.0005)
    gradient = torch.ones_like(output)
    output.backward(gradient)
    oracle.backward(gradient.double())
    for label, tensor, reference in zip(("q", "k", "v"), (query, key, value), (q64, k64, v64)):
        if label == fail_at:
            tensor.grad.add_(10)
        torch.testing.assert_close(tensor.grad.double(), reference.grad, rtol=0.005, atol=0.0005)


@pytest.mark.parametrize("fail_at,count", [(None, 5), ("q", 3), ("k", 4), ("v", 5)])
def test_diagnostic_preserves_real_assertions_and_saves_exact_state(
    monkeypatch, tmp_path, fail_at, count
):
    diagnostic = load_diagnostic(monkeypatch)
    original = torch.testing.assert_close
    row = diagnostic.diagnose_cell(helper, {"fail_at": fail_at}, tmp_path / "cell")
    assert torch.testing.assert_close is original
    assert row["status"] == ("pass" if fail_at is None else "fail")
    assert len(row["comparisons"]) == count
    assert row["comparisons"][-1]["label"] == ("v_grad" if fail_at is None else fail_at + "_grad")
    packet = torch.load(tmp_path / "cell/tensors.pt", weights_only=True)
    assert torch.equal(packet["tensors"]["query"], torch.ones(2, 1, 3, 4))
    assert torch.equal(
        packet["gradients"]["key"], torch.full((2, 1, 3, 4), 11 if fail_at == "k" else 1)
    )
    assert len(packet["block"]) == 9
    assert len(row["tensor_sha256"]["query"]) == 64
    assert row["packet_sha256"] == diagnostic.file_sha256(tmp_path / "cell/tensors.pt")
    assert (row["traceback"] is None) == (fail_at is None)


def test_diagnostic_does_not_retain_early_comparison_tensors(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)

    def lifetime_helper():
        transient = torch.ones(3)
        reference = weakref.ref(transient)
        torch.testing.assert_close(transient, torch.ones(3))
        del transient
        gc.collect()
        assert reference() is None
        raise RuntimeError("late synthetic failure")

    original = torch.testing.assert_close
    row = diagnostic.diagnose_cell(lifetime_helper, {}, tmp_path / "cell")
    assert torch.testing.assert_close is original
    assert "late synthetic failure" in row["error"]
    assert len(row["comparisons"]) == 1
    assert row["status"] == "fail"


def test_diagnostic_refuses_existing_output(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)
    with pytest.raises(FileExistsError):
        diagnostic.diagnose_cell(helper, {}, tmp_path)


def test_diagnostic_labels_precomparison_failure_as_missing(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)

    def failed_helper():
        raise RuntimeError("before any comparison")

    row = diagnostic.diagnose_cell(failed_helper, {}, tmp_path / "cell")
    assert row["status"] == "fail"
    assert row["comparisons"] == []
    assert "before any comparison" in row["error"]
    assert row["tensor_sha256"] == {}


def test_diagnostic_rejects_success_without_all_five_comparisons(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)

    def incomplete_helper():
        torch.testing.assert_close(torch.ones(1), torch.ones(1))

    row = diagnostic.diagnose_cell(incomplete_helper, {}, tmp_path / "cell")
    assert row["status"] == "incomplete"
    assert len(row["comparisons"]) == 1


def test_capture_failure_preserves_original_error_and_restores_comparator(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)

    def failed_save(*args, **kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(torch, "save", failed_save)
    original = torch.testing.assert_close
    row = diagnostic.diagnose_cell(helper, {"fail_at": "k"}, tmp_path / "cell")
    assert torch.testing.assert_close is original
    assert row["status"] == "capture_error"
    assert row["numerical_status"] == "fail"
    assert "not close" in row["error"]
    assert "synthetic disk failure" in row["capture_error"]


def test_comparator_runtime_error_is_never_recorded_as_pass(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)

    def comparator_error(*args, **kwargs):
        raise RuntimeError("synthetic comparator failure")

    monkeypatch.setattr(torch.testing, "assert_close", comparator_error)
    row = diagnostic.diagnose_cell(helper, {}, tmp_path / "cell")
    assert row["status"] == "fail"
    assert row["comparisons"] == [{"label": "grad_output", "status": "error"}]
    assert "synthetic comparator failure" in row["error"]
    assert torch.testing.assert_close is comparator_error


def test_config_validation_checks_frozen_file_and_source_hashes(monkeypatch, tmp_path):
    diagnostic = load_diagnostic(monkeypatch)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_sha256": {}, "runtime_files_sha256": {}, "expected_versions": {}})
    )
    with pytest.raises(ValueError, match="config hash mismatch"):
        diagnostic.checked_config(config, "wrong", tmp_path)
    assert (
        diagnostic.checked_config(config, diagnostic.file_sha256(config), tmp_path)["source_sha256"]
        == {}
    )
    payload = json.loads(config.read_text())
    payload["source_sha256"] = {"config.json": "wrong"}
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source hash mismatch"):
        diagnostic.checked_config(config, diagnostic.file_sha256(config), tmp_path)
