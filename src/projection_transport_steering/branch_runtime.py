import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.cache_utils import Cache


@dataclass
class BranchState:
    input_ids: Tensor
    attention_mask: Tensor
    cache: Cache | None
    generated_lengths: Tensor
    terminated: Tensor
    rng_states: tuple[Tensor, ...]


def start_branches(input_ids: Tensor, attention_mask: Tensor, seeds: list[int]) -> BranchState:
    if (
        input_ids.ndim != 2
        or input_ids.numel() == 0
        or input_ids.dtype != torch.long
        or attention_mask.shape != input_ids.shape
        or attention_mask.device != input_ids.device
        or len(seeds) != len(input_ids)
        or not all(isinstance(seed, int) and 0 <= seed < 2**63 for seed in seeds)
    ):
        raise ValueError("invalid branch batch or seed identities")
    mask = attention_mask.long()
    if (
        not torch.all((attention_mask == 0) | (attention_mask == 1))
        or not torch.all(mask[:, -1] == 1)
        or torch.any(mask[:, 1:] < mask[:, :-1])
    ):
        raise ValueError("branches require nonempty left-padded inputs")
    return BranchState(
        input_ids.clone(),
        mask.clone(),
        None,
        torch.zeros(len(input_ids), dtype=torch.long, device=input_ids.device),
        torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device),
        tuple(
            torch.Generator(device=input_ids.device).manual_seed(seed).get_state() for seed in seeds
        ),
    )


@torch.no_grad()
def advance_branches(
    model: nn.Module,
    state: BranchState,
    *,
    steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int,
) -> BranchState:
    if model.training or not isinstance(steps, int) or steps < 0:
        raise ValueError("branch continuation requires evaluation mode and a nonnegative budget")
    if not eos_token_ids or min(eos_token_ids) < 0 or pad_token_id < 0 or top_k < 0:
        raise ValueError("invalid EOS, padding, or top-k configuration")
    processors = LogitsProcessorList([TemperatureLogitsWarper(temperature)])
    if top_k:
        processors.append(TopKLogitsWarper(top_k))
    processors.append(TopPLogitsWarper(top_p))
    branch = copy.deepcopy(state)
    device = branch.input_ids.device
    generators = [torch.Generator(device=device).set_state(value) for value in branch.rng_states]
    eos = torch.tensor(eos_token_ids, device=device)
    for _ in range(steps):
        active = ~branch.terminated
        if not torch.any(active):
            break
        pending = branch.input_ids if branch.cache is None else branch.input_ids[:, -1:]
        positions = (branch.attention_mask.cumsum(-1) - 1).clamp_min(0)
        output = model(
            input_ids=pending,
            attention_mask=branch.attention_mask,
            position_ids=positions[:, -pending.shape[1] :],
            past_key_values=branch.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        branch.cache = output.past_key_values
        logits = processors(branch.input_ids, output.logits[:, -1].float())
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.full((len(active),), pad_token_id, dtype=torch.long, device=device)
        for index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            sampled[index] = torch.multinomial(
                probabilities[index], 1, generator=generators[index]
            )[0]
        branch.input_ids = torch.cat((branch.input_ids, sampled[:, None]), dim=1)
        branch.attention_mask = torch.cat((branch.attention_mask, active.long()[:, None]), dim=1)
        branch.generated_lengths += active.long()
        branch.terminated |= active & torch.isin(sampled, eos)
    branch.rng_states = tuple(generator.get_state() for generator in generators)
    return branch
