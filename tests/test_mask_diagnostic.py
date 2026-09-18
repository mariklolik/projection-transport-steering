import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import masking_utils
from transformers.masking_utils import flex_attention_mask


def load_mask_diagnostic(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_mask_diagnostic.py"
    assert path.exists(), "native mask diagnostic adapter is not implemented"
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("mask_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mask_helper(mutate=False, fail=False, calls=1):
    for _ in range(calls):
        block = flex_attention_mask(2, 3, 3, device="cpu")
    if mutate:
        block.kv_num_blocks.add_(10)
    if fail:
        raise RuntimeError("synthetic helper failure")
    for _ in range(5):
        torch.testing.assert_close(torch.ones(1), torch.ones(1))


@pytest.mark.parametrize("policy,effective", [("automatic", True), ("eager", False)])
def test_policy_changes_only_native_compile_argument(monkeypatch, policy, effective):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    observed = []

    def cpu_builder(**kwargs):
        observed.append(dict(kwargs))
        return native(**{**kwargs, "_compile": False})

    monkeypatch.setattr(masking_utils, "create_block_mask", cpu_builder)
    monkeypatch.setattr(module, "create_block_mask", cpu_builder)
    padding = torch.tensor([[False, True, True, True], [True, True, False, True]])
    with module.mask_construction(policy, capture=True) as snapshots:
        block = flex_attention_mask(2, 2, 4, q_offset=2, attention_mask=padding, device="cpu")
    assert masking_utils.create_block_mask is cpu_builder
    assert len(observed) == len(snapshots) == 1
    assert observed[0]["_compile"] is effective
    assert {k: v for k, v in observed[0].items() if k != "mask_mod"} == {
        "B": 2,
        "H": None,
        "Q_LEN": 2,
        "KV_LEN": 4,
        "device": "cpu",
        "_compile": effective,
    }
    assert snapshots[0]["requested_compile"] is True
    assert snapshots[0]["effective_compile"] is effective
    for batch in range(2):
        for query in range(2):
            for key in range(4):
                actual = block.mask_mod(
                    torch.tensor(batch), 0, torch.tensor(query), torch.tensor(key)
                )
                assert bool(actual) == (query + 2 >= key and bool(padding[batch, key]))


def test_eager_uses_real_native_builder_without_capture(monkeypatch):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    with module.mask_construction("eager") as snapshots:
        block = flex_attention_mask(2, 3, 3, device="cpu")
    assert snapshots == []
    assert masking_utils.create_block_mask is native
    assert block.kv_num_blocks.flatten().tolist() == [1, 1]


def test_native_builder_exception_restores_binding(monkeypatch):
    module = load_mask_diagnostic(monkeypatch)

    def broken(**kwargs):
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(masking_utils, "create_block_mask", broken)
    monkeypatch.setattr(module, "create_block_mask", broken)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        with module.mask_construction("eager"):
            flex_attention_mask(2, 3, 3, device="cpu")
    assert masking_utils.create_block_mask is broken


def test_unknown_policy_is_rejected_before_patch(monkeypatch):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    with pytest.raises(ValueError, match="mask policy"):
        with module.mask_construction("unregistered"):
            pytest.fail("invalid policy entered")
    assert masking_utils.create_block_mask is native


@pytest.mark.parametrize("mutate", [False, True])
def test_late_packet_and_early_metadata_are_distinct_snapshots(monkeypatch, tmp_path, mutate):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    kwargs = {"mask_policy": "eager", "mutate": mutate}
    row = module.diagnose_mask_cell(mask_helper, kwargs, tmp_path / "cell")
    assert kwargs == {"mask_policy": "eager", "mutate": mutate}
    assert masking_utils.create_block_mask is native
    assert row["status"] == row["mask_capture_status"] == "pass"
    metadata = json.loads((tmp_path / "cell/mask-metadata.json").read_text())
    assert metadata["before"][0]["block"]["kv_num_blocks"] == [[[1]], [[1]]]
    assert metadata["after"]["kv_num_blocks"] == [[[11 if mutate else 1]], [[11 if mutate else 1]]]
    assert len(row["comparisons"]) == 5
    assert row["mask_metadata_sha256"] == module.file_sha256(tmp_path / "cell/mask-metadata.json")


def test_helper_failure_keeps_original_error_and_restores_builder(monkeypatch, tmp_path):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    row = module.diagnose_mask_cell(
        mask_helper, {"mask_policy": "eager", "fail": True}, tmp_path / "cell"
    )
    assert row["status"] == "fail"
    assert row["mask_capture_status"] == "pass"
    assert "synthetic helper failure" in row["error"]
    assert masking_utils.create_block_mask is native


@pytest.mark.parametrize("calls", [0, 2])
def test_missing_or_duplicate_construction_never_passes_capture(monkeypatch, tmp_path, calls):
    module = load_mask_diagnostic(monkeypatch)
    row = module.diagnose_mask_cell(
        mask_helper, {"mask_policy": "eager", "calls": calls}, tmp_path / "cell"
    )
    assert row["status"] == "mask_capture_error"
    assert row["numerical_status"] == "pass"
    assert "exactly one" in row["mask_capture_error"]


def test_unexpected_helper_mask_binding_is_rejected(monkeypatch, tmp_path):
    module = load_mask_diagnostic(monkeypatch)

    def other_helper():
        return None

    monkeypatch.setitem(other_helper.__globals__, "flex_attention_mask", object())
    with pytest.raises(ValueError, match="helper mask binding"):
        module.diagnose_mask_cell(other_helper, {"mask_policy": "eager"}, tmp_path / "cell")


def test_non_native_builder_binding_is_rejected(monkeypatch):
    module = load_mask_diagnostic(monkeypatch)
    other = object()
    monkeypatch.setattr(masking_utils, "create_block_mask", other)
    with pytest.raises(ValueError, match="native builder binding"):
        with module.mask_construction("eager"):
            pytest.fail("non-native binding entered")
    assert masking_utils.create_block_mask is other


def test_unexpected_compile_request_is_rejected_and_restored(monkeypatch):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask
    with pytest.raises(ValueError, match="compile request"):
        with module.mask_construction("eager"):
            masking_utils.create_block_mask(_compile=False)
    assert masking_utils.create_block_mask is native


def test_early_capture_exception_preserves_helper_error(monkeypatch, tmp_path):
    module = load_mask_diagnostic(monkeypatch)
    native = masking_utils.create_block_mask

    def broken_snapshot(block):
        raise RuntimeError("synthetic snapshot failure")

    monkeypatch.setattr(module, "block_metadata", broken_snapshot)
    row = module.diagnose_mask_cell(mask_helper, {"mask_policy": "eager"}, tmp_path / "cell")
    assert row["status"] == "mask_capture_error"
    assert row["numerical_status"] == "fail"
    assert "synthetic snapshot failure" in row["error"]
    assert masking_utils.create_block_mask is native
