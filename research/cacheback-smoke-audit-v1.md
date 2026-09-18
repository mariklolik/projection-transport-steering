# CacheBack one-state geometry smoke audit, version 1

Execution date: 2026-09-06. Status: exposed implementation smoke only; no sentinel, development, pilot-test, behavioral, or promotion claim.

## Purpose and setup

The smoke tested whether exact full-vocabulary, float64 future pullback geometry is operational on the pinned GPT-2 Small checkpoint before materializing the frozen data packet. It used one manually supplied neutral prompt, layer 6, one sampled two-token continuation, horizons 0 and 2, and the existing third-person morphology covector. The prompt SHA-256 was `f93f34f7319bb3144e656746a2fb42b396fafdf365985f06ee8f28e54568d10c`; generated token IDs were `[257,2176]`.

Host `avi-gn-fsk40` used one NVIDIA H100 80GB HBM3. Source hashes were `da6807fa96fba69c76fc2f05553d617b22b2f0d7375f5eb0bb1bb381b48703a7` for `future_pullback.py` and `5a52ab32be0be2cc466f6d0d908e948bc26450a4fe087329549885b47bc67143` for `claim_relative_geometry.py`. Pilot-test states observed: zero.

## Results

- Exact three-offset geometry took 0.611 seconds and peak allocated GPU memory was 3,436,961,280 bytes.
- The minimum eigenvalue of `G_2-G_0` was `3.62e-19`, consistent with positive-semidefinite nesting at smoke precision.
- The action cosine between `H=0` and `H=2` was `0.78682`, so the future metric was not a numerical duplicate of the immediate metric on this state.
- At local target `rho=0.05`, realized immediate semantic effects were `0.05066` for `H=0` and `0.05016` for `H=2`.
- Cumulative teacher-forced forward KL was `1.0812e-5` for `H=0` and `1.4004e-6` for `H=2`. Per-offset values were `[9.68e-7,5.43e-6,4.42e-6]` and `[1.20e-6,1.83e-7,1.98e-8]`.

## Interpretation

This is a useful feasibility signal: exact future geometry is cheap enough to justify the frozen sentinel, and the selected state exhibits a large action-angle and KL difference. It is not an effect estimate because the prompt was manually chosen, there is one independent state, calibration is local rather than exactly matched in realized effect, no baselines beyond `H=0` were run, and no cache-reset/cache-only diagnostic exists yet. The 87% raw KL difference must not be quoted as a method result.

The smoke also emitted a generation attention-mask warning because the tokenizer's pad token equals EOS. The screening and future runners must pass the explicit attention mask; no smoke rerun is needed because there was no padding and this result is not admitted as evidence.

Decision: continue only to frozen source-data materialization and cache-path implementation tests. Do not open the untouched pilot or scale models.
