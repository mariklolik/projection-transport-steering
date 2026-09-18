# Prospective fixed-baseline resource amendment v2

Date: 2026-09-07 18:17:39 UTC. Scope: the unopened IRC fixed-adapter fit/selection stage only. This is an explicit plan amendment under the user's existing request for autonomous research on the available FSK GPUs. It does not claim that the user supplied or specifically approved a numerical budget.

## Decision and authority

Replace the assistant-origin **8 H100-hour baseline-stage ceiling with 48 H100-hours**, before any eligible baseline checkpoint or scientific selection output exists. The original v1 protocol and its failed engineering packets remain immutable historical records. No result may be described as achieved within the original eight-hour envelope.

The goal objective at `/Users/mekashirskiy/.codex/attachments/322808f2-1bda-44c4-bb66-c9855b21a266/goal-objective.md` (SHA256 `522da7bfa67be76f8d7919823ca1e09b613626a2d49e9d1bbbebd9e7642f118a`) authorizes long-running FSK research but contains no numerical compute ceiling. This amendment uses the same host and at most its existing eight GPUs; it does not authorize purchases, other infrastructure, interference with unrelated jobs, or an unlimited run.

The one-hour engineering ceiling is unchanged. Its latest measured remainder is 70.272500518 seconds after the separately predeclared static FP64 test. Engineering expenditure is not moved into this amended baseline budget. The downstream response-pilot, response-training and development ceilings of 12/36/16 H100-hours also remain unchanged. Their complete packets need fresh conditional cost review before launch; this amendment does not imply that they will fit.

## Evidence and reason

`irc-baseline-cost-scenarios-v1.json` extrapolates eight 12-epoch fits at approximately 10.21–11.17 H100-hours using source-only native length grouping. Applying the completed 16-question unmodified runtime to eight candidates plus zero on all 256 selection questions gives 26.4269 H100-hours. The combined 36.64–37.60-hour scenario motivates a 48-hour bounded envelope, roughly 25% above the larger interpolation.

This is a planning sensitivity, **not a measured full run, prediction interval or safe upper bound**. Next-bucket sensitivity is materially larger; actual training accumulation, other ranks/sites, compiled-shape behavior and steered rollout lengths can invalidate the interpolation. An enforced deadline still closes an incomplete stage; it does not make the cost estimate true.

Independent read-only review established that the full original stage cannot currently be admitted as affordable within eight hours, while also rejecting a claim of irreducible impossibility. Successive halving would be a different exploratory selection contract, not complete evaluation of the frozen family. This amendment keeps complete candidate coverage instead.

## Unchanged scientific contract

- All 1,500 fit questions and exact public solutions; no generated labels, truncation or source exclusions.
- All 256 immutable selection questions, common per-question sampling seeds, one complete rollout per condition, unchanged 8,192-token cap and evaluator.
- Eight candidates: ranks 4/8 × post-block sites 17/23 × prompt/full application. Both policies are trained separately.
- One final checkpoint per candidate after the source-derived 12-epoch recipe; no intermediate-checkpoint selection or hidden early stopping.
- All four full and four prompt-only candidates receive the complete selection allocation. The shared zero-action reference is evaluated once per question, not once per candidate.
- The learned adapter is evaluated at its native unit strength, `g = 1`. This fixes the previously unresolved strength schedule prospectively; no extra strength, learning-rate, epoch or checkpoint search is permitted in this stage.
- The best full-generation adapter and the best prompt-only adapter are retained separately. Selection maximizes exact correctness; ties prefer lower rank, then the lower site index. A full-generation candidate must strictly exceed the shared zero-action selection mean before response collection can open. A prompt-only winner cannot replace this entry condition.
- Conditional on that entry pass, `g_ref = 1` and the already declared response actions are `{0, 0.5, 1}`. The selected adapter is exploratory, not a confirmed gain or a strongest-overall baseline.

The eight-candidate family is a restricted action initializer, not an exact reproduction of all-layer ReFT or proof against LoRA/RISER/ACTS. All stronger-comparator, mechanism-control, uncertainty and three-family/three-construct obligations remain unchanged. Future competitors must receive a reported matched resource/data envelope; this amendment cannot buy an unreported advantage for IRC.

## Execution and accounting

Use at most eight resident workers, one candidate per GPU, after fresh idle-device checks. Load each base model once where safe. Partition the shared zero-action references deterministically across the workers. Keep every candidate's 256 rows, final adapter artifact, optimizer/seed/data/source receipt and technical failures. Validate completed chunks on CPU without holding the GPU queue.

The stage's total container lifetime is capped at 48 H100-hours, including model loading, compilation, actual fitting, saving, all selection generations, scoring inside containers, failed jobs and restart attempts. Separate waiting or CPU-only work is recorded but is not invented as accelerator use. Each initial GPU worker has at most six hours including termination grace; any retry draws from unused stage allowance and needs a recorded technical-failure decision. There is no automatic numerical or unfavorable-result retry.

Before dispatch, freeze the exact source hashes, accumulation/token-weighting contract, optimizer/scheduler defaults, finite padding schedule, checkpoint format, row ordering and per-worker deadline. Require a current numerical engine receipt, native accumulated-update test and saved-adapter generation contract. The static-engine FP64 check alone does not satisfy those dependencies.

No partial subset may be labeled a completed comparison. If the whole stage exceeds the amended envelope, retain all observed outputs and label the stage `budget-incomplete`; do not declare omitted candidates defeated or alter the cap, questions, methods, seeds or primary endpoint.

## Research decision and remaining risk

The primary unresolved scientific question is still whether repeated response-informed control beats a useful fixed action under matched resources. This amendment resolves an unsupported planning allowance, not that scientific question. Failed v20, observer and scalar-trajectory routes remain closed or failed at their exact prior scope.

The next implementation increment is the existing fit helper's target-token-normalized accumulation to effective batch 32, followed by a bounded native qualification and exact full-stage dispatch receipt. Final GPU launch admission remains pending those checks. No SOTA, useful checkpoint, positive action or submission-readiness claim follows from this plan.

Responsible-author verification and AI-use disclosure classification remain unverified.

