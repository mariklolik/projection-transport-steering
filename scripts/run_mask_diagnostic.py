import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import masking_utils

import run_attention_diagnostic as diagnostic
from run_attention_diagnostic import diagnose_cell, file_sha256


def block_metadata(block) -> dict:
    result = {"shape": list(block.shape), "BLOCK_SIZE": list(block.BLOCK_SIZE)}
    for prefix in ("kv", "q", "full_kv", "full_q"):
        for suffix in ("num_blocks", "indices"):
            name = f"{prefix}_{suffix}"
            tensor = getattr(block, name)
            result[name] = None if tensor is None else tensor.detach().cpu().tolist()
    return result


@contextmanager
def mask_construction(policy: str, *, capture: bool = False):
    if policy not in ("automatic", "eager"):
        raise ValueError("unknown mask policy")
    native = masking_utils.create_block_mask
    if native is not create_block_mask:
        raise ValueError("unexpected native builder binding")
    snapshots = []

    def build(**kwargs):
        if kwargs.get("_compile") is not True:
            raise ValueError("unexpected native compile request")
        if policy == "eager":
            kwargs["_compile"] = False
        block = native(**kwargs)
        if capture:
            snapshots.append(
                {
                    "requested_compile": True,
                    "effective_compile": kwargs["_compile"],
                    "block": block_metadata(block),
                }
            )
        return block

    with patch.object(masking_utils, "create_block_mask", build):
        yield snapshots


def diagnose_mask_cell(helper, kwargs: dict, output: Path) -> dict:
    if helper.__globals__.get("flex_attention_mask") is not masking_utils.flex_attention_mask:
        raise ValueError("unexpected helper mask binding")
    options = dict(kwargs)
    policy = options.pop("mask_policy")
    with mask_construction(policy, capture=True) as snapshots:
        row = diagnose_cell(helper, options, output)
    metadata = {"policy": policy, "before": snapshots, "after": None, "error": None}
    row["mask_capture_status"] = "pass"
    try:
        if len(snapshots) != 1:
            raise ValueError("expected exactly one native mask construction")
        packet = torch.load(output / "tensors.pt", map_location="cpu", weights_only=True)
        if packet["block"] is None:
            raise ValueError("post-helper block metadata is missing")
        metadata["after"] = block_metadata(
            SimpleNamespace(shape=packet["block_shape"], **packet["block"])
        )
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        row.update(
            status="mask_capture_error",
            mask_capture_status="fail",
            mask_capture_error=metadata["error"],
        )
    path = output / "mask-metadata.json"
    with path.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    row["mask_metadata_sha256"] = file_sha256(path)
    return row


if __name__ == "__main__":
    with patch.object(diagnostic, "diagnose_cell", diagnose_mask_cell):
        diagnostic.main()
