# Latent Reward Steering source audit, version 1

Audit date: 2026-09-07. Status: source-complete preflight for comparator admission.

## Pinned source

- Paper: [Latent Reward Steering, arXiv:2606.00726v3](https://arxiv.org/abs/2606.00726), reported as accepted to EMNLP 2026.
- Repository: [jiakanglee/Latent-Reward-Steering](https://github.com/jiakanglee/Latent-Reward-Steering), commit `5becfd7b2fa9ae80bab713d4a93dbcb36d146fe6`.
- Collection script SHA-256: `48cd70c29fc7522c9d69190f78ff57d730aee5000d225bdb42fb12bec15a0c07`.
- Reward-training script SHA-256: `b39557c30b4a5312205c77c6f91fe2b6c366ed705a8596bdc6c3d0a9b5564102`.
- Steering script SHA-256: `1b73c95d73c3af038f700fa45429028d0acff96aa4c7a537f0c6883b7370344a`.
- README SHA-256: `ec9301208fcd367684921c68679124f19be556d4ace5adf5bec92fc72b6f8082`.

## What the paper owns

LRS labels complete reasoning traces by final-answer correctness, trains a Transformer reward model over SAE-latent sequences, and at inference uses the reward gradient at each generated token. A reward-confidence gate limits interventions. The paper reports six zero-shot reasoning benchmarks on Open-Reasoner-Zero 7B and 1.5B, including `+4.4` on MATH-500 and `+10.0` percentage points on AIME 2024 and AIME 2025 for the 7B model. These values are paper-specific frontier markers, not common-protocol SOTA thresholds.

This closes novelty claims based only on terminal-reward supervision, state-dependent reward gradients, SAE-space correction, confidence gating, or selective token intervention.

## Reproduction findings

The exact public pipeline is not presently executable from the pinned repository:

1. `python3 -m py_compile LRS/collect_data/generate_data_7B.py` fails at line 84 on a full-width comma `U+FF0C`; another instance occurs at line 107.
2. No `.pt`, `.safetensors`, or `.bin` SAE or reward-model artifact is present in the repository. The runtime expects local artifacts, including `transformer_reward_model_aime_best.pt`.
3. The documented AIME pipeline collects traces from AIME 2024 and AIME 2025 together, trains the reward model on that combined file, and then evaluates on AIME 2024 and AIME 2025. No held-out question split is declared in the inspected README or training source.
4. The training script constructs one sampler and one training loader. It has no validation loader or split; the `_best.pt` checkpoint is selected by the lowest epoch training loss.
5. The inference repository contains dataset-model-specific sweeps. The published configurations vary `K`, step size, and thresholds by model-dataset cell, including separate AIME 2024 and AIME 2025 settings.
6. The reward model is trained on sequences but the steering hook passes a sequence of length one at deployment. The paper acknowledges the token-local query as a limitation.

These findings do not refute the paper's reported results. They prevent an exact released-code reproduction from serving as a fair-strong comparator receipt.

## Comparator decision

Two records remain separate:

- `published LRS`: cite the paper's values only as unmatched frontier markers;
- `LRS-like common protocol`: clean-room normalized Euclidean reward-gradient steering, using the same prefix-value model, data, layer, gate budget, evaluation questions, decoding, and compute accounting as the candidate.

The clean-room baseline is not described as an exact LRS reproduction. It must not copy broken source, inherit AIME 2024/2025 training overlap, tune per evaluation dataset, or use unavailable weights.

No GPU run is justified until the common protocol, exact scorer, immutable splits, baseline identities, and promotion rules are frozen.
