# OLMo-2 gated control freeze, version 17

Freeze date: 2026-09-05. This registry is frozen after OLMo-2 passed the version 16 candidate-versus-Spherical gate and before any OLMo-2 control output.

The fixed candidate, model revision, layer, representation-fit groups, observer groups, 237 confirmatory groups, prompt, answer scoring, intervention span, candidate temperature, and candidate outputs remain exactly those of method version 10 and benchmark version 16.

The family contains seven controls:

- no intervention;
- five isotropic directions drawn with CPU PyTorch seeds `2026090501` through `2026090505`, each normalized and assigned the exact per-state update norm produced by DAPS;
- a deployment-misaligned ablation fit on the same 300 representation-fit questions using `Q: {question} A: {answer}` without the evaluation instruction, only the final answer-token state, truthful-minus-false unit mean contrast, balanced L2 logistic posterior with `C=1`, and the unchanged hard spherical action at `alpha=1,rho=0.5`.

Each control receives a scalar temperature fit only on the same 160 observer groups, then is evaluated once on the same 237 confirmatory groups. Candidate-minus-control paired calibrated-MC2 and MC1 BCa intervals use 10,000 resamples, seed `20260905`, and Bonferroni two-sided alpha `0.05/7`. The specificity stage passes only if every calibrated-MC2 lower bound is above zero and every MC1 lower bound is above `-0.015`. ID and shifted strata are descriptive and use unadjusted 95% intervals.

No control result authorizes a claim across all three architecture families. It tests only whether the confirmed OLMo-2 effect survives the predeclared magnitude and fit-alignment controls.
