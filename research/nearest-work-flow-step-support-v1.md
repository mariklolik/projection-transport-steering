# Flow-step support nearest-work audit, version 1

Audit date: 2026-09-07. Status: bounded primary-source search complete; mechanism sentinel only.

## Candidate boundary

The internal candidate is a fixed temporal-support intervention over the released three-step FLAS operator. It does not train a model, add a residual vector, choose a per-prompt gate, or tune flow time. The candidate suppresses the first Euler velocity evaluation, retains the released evaluations at step indices 1 and 2, and rescales their net displacement by `3/2` so the nominal integrated update mass equals the released three-step path. This is an intervention on the support of the learned vector field, not a valid lower-step numerical approximation to the original ODE.

## Direct ownership

| Work | Owned contribution | Boundary here |
|---|---|---|
| FLAS, arXiv:2605.05892 and commit `720ef8a67697d9b94130b374b5b3a1522a782566` | Learned concept-conditioned vector field, flow time, multi-step Euler integration, per-token trajectories, step-count ablations, and the empirical observation that early directions differ from mutually aligned late directions | Every learned component and the early-versus-late geometric observation belong to FLAS. Its released `_iv` hook already supports selecting step indices for analysis. Only a prospectively tested fixed support rule and causal subset decomposition could be new evidence |
| ODESteer, ICLR 2026; FlowSteer; TruthFlow; K-Steering | Multi-step state-dependent steering dynamics | Multi-step steering, ODE interpretation, barrier dynamics, and task-specific flows cannot be claimed |
| Dynamic Activation Composition, DSAS, GAPS, CAST, CLAS, StTP/StMP, GCAD | Tokenwise, coordinatewise, context-conditioned, or feedback-controlled intervention strength and gating | Adaptive strength, selective steering, token gating, and online feedback cannot be claimed; this sentinel uses one fixed concept-independent support |
| UniSteer, arXiv:2605.30076 | Conditional source inversion and target flow transport | Flow inversion and source-target composition cannot be claimed |
| Generic ODE pruning and numerical integration | Step removal, solver approximation, and equal-compute integration | This is not presented as a numerical solver advance or faithful approximation theorem |

## Search result

The primary FLAS paper reports full-step-count ablations at `N` in `{1,2,3,4,5,10}` and velocity cosine/magnitude analysis, but no performance or safety table for causal subsets of a fixed trained `N=3` checkpoint. The released code exposes subset selection as an analysis intervention, which makes method novelty especially uncertain. Exact-keyword searches found no language-model activation-steering paper that reports an exact all-subset decomposition of a fixed learned intervention flow and prospectively evaluates a globally fixed late-support deployment rule. This absence is only a bounded search result, not a novelty claim.

## Gate consequence

The next experiment is a cheap mechanism sentinel, not a paper-method promotion. If the fixed late-support candidate fails to Pareto-improve unchanged FLAS and beat the matched early-support control, the route closes. A pass would justify a fresh-development test and a deeper novelty audit, not contribution wording.
