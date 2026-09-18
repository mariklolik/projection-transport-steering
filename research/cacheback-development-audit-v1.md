# CacheBack development audit, version 1

Date: 2026-09-06. Decision: close the powered CacheBack performance route under the frozen coverage gate. Retain only the local future-metric and cache-path mechanism result. Do not open the 48 pilot states.

## Packet and execution

The accepted packet is `artifacts/development/cacheback_v1_development_attempt2`. It contains all 72 distinct state-layer cells from 24 independent development states and layers 3, 6, and 9. Every cell contains all nine frozen dose-by-regularization configurations and six required methods, for 648 configurations and 3,888 method evaluations. All eight shard receipts report 9/9 passed rows, forward-mode AD, zero VJPs, and zero pilot access. The packet contains 3,981,312 derivative rows and 442,368 JVP directions. Shard wall times range from 146.53 to 181.16 seconds and maximum peak allocation is 21.67 GB.

The first launch is preserved in `cacheback_v1_development_attempt1_invalid_cpu_oversubscription`. Eight workers each inherited about 40 CPU threads, causing severe host oversubscription while GPUs were intermittently idle. Only the eight verified development PIDs were stopped. The accepted launch fixes PyTorch intra-op threads at 2 and inter-op threads at 1 per worker; CPU use fell to about 1.3 to 1.5 cores per process while GPU utilization reached 43% to 100% during sampled checks. No method, state, layer, dose, regularization, or threshold changed.

## Frozen selection result

The prospective rule requires one dose for which every required method reaches the matched immediate effect on at least 80% of states overall and within each concept. With eight states per concept, a concept stratum therefore needs at least 7/8 reached states.

No layer-by-regularization cell has an eligible common dose. The best worst-stratum coverage is 5/8, or 62.5%, at layer 3 with regularization multiplier 1.0 for doses 0.125 and 0.25. At dose 0.125, CAA reaches 5/8 past-tense states; CacheBack H2, H4, H8, and Euclidean each reach only 5/8 third-person states. At dose 0.25, H2, H4, H8, and Euclidean again reach only 5/8 third-person states. Dose 0.5 falls to 4/8 for Euclidean on third-person states.

The failure is not a missing-output artifact. Many FishBack and CacheBack failures are finite-action endpoint shortfalls of roughly 0.2% to 3% below the requested effect. Euclidean shortfalls reach about 1.6% in the inspected limiting cells. CAA is substantially less regular: some locally matched directions produce a finite realized effect below half the target, and a few flip the semantic-effect sign. The frozen `[0,1]` scale bracket and no-extrapolation rule correctly count these cases as unreachable.

Because no common dose exists, no layer or regularization is selected. Primary contrasts, powered cache mediation, free-generation diagnostics, and the pilot are not authorized. Enlarging the scale bracket, relaxing the tolerance, dropping Euclidean or CAA, pooling concepts, or selecting only reachable states would be a post-development rescue and is prohibited.

## Diagnostic signal on reachable subsets

The analyzer preserves diagnostics for every cell and dose, but these subsets are selected by reachability and cannot support the frozen performance claim.

For layer 6, regularization multiplier 0.1, and dose 0.125, 20/24 states have matched H0 and each future horizon. Mean cumulative-KL reductions are 88.74%, 93.39%, and 94.98% for H2, H4, and H8. Their Holm-ordered one-sided BCa lower bounds are 82.96%, 89.65%, and 91.86%, and all Holm-adjusted signed-rank p-values against the 15% threshold are `2.86e-6`. Immediate-effect-retention lower bounds are at least 99.97%; offset-concentration upper bounds are at most 0.584; and action-angle lower bounds are at least 0.720 radians.

Within the same cell, the state-level predicted-versus-realized Spearman mean is 0.9895 with BCa interval `[0.9791,0.9947]`. The smallest-dose median Taylor relative error is 4.65% with BCa interval `[3.97%,5.53%]`. Across all 27 diagnostic cell-dose combinations, the smallest horizon-reduction lower bound is 31.05%, the largest offset-concentration upper bound is 0.665, and the smallest retention lower bound is 99.94%. These results show that the local quadratic metric is informative conditional on reachability; they do not repair the failed coverage estimand.

## Claim boundary

The admissible result is narrow: on exposed GPT-2 morphology states, future pullback Fisher changes the local action, predicts small-dose sequence KL, and the altered-token cache carries almost all measured future KL for the matched exposed H8 action. The data do not establish a deployable method at the frozen coverage level, a powered cache-mechanism effect, behavioral improvement, architecture transfer, superiority to GCAD, or a steering-frontier result.

Pilot B is closed. Pilot A InvariantBack remains independent and may proceed from its own sentinel without using CacheBack results for selection.
