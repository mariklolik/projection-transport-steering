from typing import Any

import torch
from torch import Tensor


def matched_linear_action(direction: Tensor, covector: Tensor, target: float) -> Tensor:
    if (
        direction.ndim != 1
        or covector.shape != direction.shape
        or not torch.all(torch.isfinite(direction))
        or not torch.all(torch.isfinite(covector))
        or not target > 0
    ):
        raise ValueError("direction, covector, or target is invalid")
    progress = covector @ direction
    if progress.abs() <= torch.finfo(progress.dtype).eps:
        raise ValueError("direction cannot reach the local target")
    return target * direction / progress


def caa_direction(
    model: Any,
    mapping: dict[str, Any],
    layer: int,
    batch_size: int,
) -> Tensor:
    pairs = mapping["pairs"]
    if not pairs or batch_size < 1 or not 0 <= layer < len(model.transformer.h):
        raise ValueError("mapping, layer, or batch size is invalid")
    device = next(model.parameters()).device
    differences = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        token_ids = [[pair["base_id"]] for pair in chunk] + [
            [pair["target_id"]] for pair in chunk
        ]
        captured = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output if isinstance(output, Tensor) else output[0])

        handle = model.transformer.h[layer].register_forward_hook(capture)
        try:
            with torch.no_grad():
                model(torch.tensor(token_ids, device=device), use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("CAA activation capture failed")
        representations = captured[0][:, -1]
        differences.append(representations[len(chunk) :] - representations[: len(chunk)])
    direction = torch.cat(differences).mean(0)
    if not torch.all(torch.isfinite(direction)) or direction.norm() == 0:
        raise RuntimeError("CAA direction is invalid")
    return direction
