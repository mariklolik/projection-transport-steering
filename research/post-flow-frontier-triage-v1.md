# Post-flow frontier triage, version 1

Coverage date: 2026-09-07. Status: primary-paper triage after Flow-step support version 1 closure.

## Evidence-driven boundary

The completed all-subset sentinel shows that removing FLAS step 0 reduces neutral damage but loses efficacy, while restoring nominal mass recovers efficacy by recreating and exceeding the damage. This does not by itself identify a new controller.

Three direct neighbors block an immediate optimization or routing successor:

- [A-LQR, arXiv:2604.19018](https://arxiv.org/abs/2604.19018) already models transformer layers as a locally linear time-varying system, uses Jacobians and Riccati feedback to track adaptive semantic setpoints, and reports multi-model behavior control.
- [Activation Steering for Chain-of-Thought Compression, arXiv:2507.04742](https://arxiv.org/abs/2507.04742) already derives an output-Jacobian and curvature-based forward-KL bound for steering strength and applies it across 7B, 8B, and 32B reasoning models.
- [Deployable Per-Instance Multi-Layer Activation Steering, arXiv:2608.08829](https://arxiv.org/abs/2608.08829) already studies exhaustive and greedy per-instance support, exact subset Shapley structure, prompt-only support prediction, direction inference, and adaptive stopping on two 8B models and six traits.

The internal distinction that these papers act over transformer depth while FLAS support acts over integration time is real but too narrow to justify another GPU route. A time-local Fisher filter, per-concept support selector, KL-calibrated scale, or feedback step controller would combine already owned components and would also violate the closed version-1 no-rescue boundary unless it established a different scientific object and direct comparisons.

## Next estimand

The next efficient question is whether the teacher-forced likelihood screen used by the recent funnels predicts open-generation behavior at all. This is not a steering method and cannot advance any observed condition. It is a validation study of the research instrument.

Use the 12 already exposed concepts and all 14 already evaluated conditions. Generate the eight existing evaluation prompts per concept with fixed greedy decoding, score every output with the exact local Qwen3-14B AxBench-format proxy, and compare pre-existing teacher-forced rankings with generated concept, instruction, fluency, and harmonic-mean rankings. Freeze all correlations, agreement metrics, uncertainty, failure rules, and the no-selection boundary before generation.

A positive result only permits teacher-forced screening for future distinct hypotheses. A negative result retires likelihood-first method iteration and requires every successor to start with behavior-level generation. Neither result validates the local judge against GPT-4o-mini, supports Flow-step support, or opens fresh concepts, models, official judging, or SOTA claims.
