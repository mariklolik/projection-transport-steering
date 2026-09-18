# Actual-weight fit qualification and cost decision

2026-09-07. DRAFT_FROM_EVIDENCE audit and research plan, not submission prose. The reader decision is whether fitting and selecting the fixed action family is now numerically executable and affordable.

## Completed qualification

The prospective18-step packet passes on the pinned Qwen3-8B/H100 environment. All18 step files, the receipt and emitted JSON log events match; all47 declared source hashes match. The independent critic checked224 assertions without a material defect. Config SHA256 is `2230a7cd385c23dc74ee29aff9cde63b3e736f57ddd44040045c7907472e2e29`; result audit is `76434c32fb170987677c23ca50ca1dfcdd2c31a2e8e085c72967dfd16d0f64f0`; raw receipt is `1e5b0305730d4b19ab2620ad6f221626000dc6cf4ed5c07cb8a0dded12ea5ca9`.

Both policies update exactly the three upstream LoReFT parameter tensors, totaling65544 trainable values: FP32 rotation and BF16 learned source/bias. Each step has finite nonzero gradients and a finite changed adapter; no base gradient or base parameter-version change is observed. These ownership/version checks do not cryptographically compare every base-weight byte. No checkpoint is baseline-eligible.

| Policy | Padded pair shape | First-in-cell seconds | Next two step seconds | Supervised tokens per step | Active input positions |
|---|---|---:|---|---:|---:|
| prompt | 2 × 384 | 17.456 | 0.150, 0.146 | 366 | 28 |
| prompt | 2 × 896 | 19.249 | 1.446, 1.443 | 1274 | 28 |
| prompt | 2 × 2816 | 11.695 | 11.665, 11.670 | 4327 | 28 |
| full | 2 × 384 | 0.381 | 0.378, 0.377 | 366 | 394 |
| full | 2 × 896 | 1.449 | 1.446, 1.445 | 1274 | 1302 |
| full | 2 × 2816 | 11.699 | 11.663, 11.676 | 4327 | 4355 |

These are six length-selected fit questions and sequential updates in one process, not18 independent trials. First-in-cell includes any applicable compilation; the second policy reuses caches. The losses are retained, including the initial prompt-median increase1.2232→1.2572, but loss trends do not select a recipe or establish behavioral improvement. No confidence interval or performance comparison is inferred from these timings.

The complete container takes138.9011713 seconds; the runner takes124.9689653 and measured step intervals sum to115.4336766. The cumulative post-load allocated-memory maximum is44111876096 bytes, or41.0824 GiB; it is not isolated per-cell memory or all resident GPU memory. Known measured runtime-stage cost is0.8200937088 H100-hours, with a separate0.10-hour historical-overhead reserve. This is not an exact lifetime bill.

## Full-source cost reconstruction

All1500 original fit targets retain507475 total and356743 supervised tokens per epoch. The ending audit hashes records in original packet order; sorting those same records by length produces a different content hash. The first cost-analysis comparison mistakenly mixed these orders and is retained in `irc-fit-batching-cost-inputs-attempt1.json`. The corrected exact remote CPU reconstruction matches the original `54ab81b1...` hash without changing a source or token.

`irc-fit-batching-cost-inputs-v1.json` reconstructs12 seed-fixed epoch schedules using existing Transformers length grouping and three planning alternatives. Random pairs require9364992 padded token instances; the native length-grouped pairing requires7436544,20.5921% fewer. Its attention-square proxy is30.3242% lower. These are deterministic workload counts, not measured GPU speedups. Pair ordering changes effective optimizer batches unless accumulation is separately specified and tested. Globally sorted schedules are planning references, not an admitted curriculum.

`irc-baseline-cost-scenarios-v1.json` makes the budget gap explicit. Interpolating the observed pair timings in length or squared length gives roughly10.2–11.2 H100-hours for eight12-epoch candidates with native length grouping, before orchestration or selection. This is a sensitivity calculation across unmeasured shapes/ranks/sites, not a statistical estimate or safe upper bound. Applying one unmodified16-question runtime to nine256-question conditions gives26.4269 hours of naive generation cost; trained-hook overhead and the selection length distribution are unmeasured. Neither calculation authorizes launch or proves an irreducible cost.

## Decision and next action

The numerical fit prerequisite improves; scientific effect S5 and strongest-comparator parity S7 remain0 and the paper remains EVIDENCE_THIN. Do not launch the eight-way grid or declare the finite family exhausted from a cost interpolation.

The one next diagnostic is `irc-fit-profile-config-v1.json`: full policy, rank8/site17, the same p95/maximum pairs, two sequential steps per pair, with CPU/CUDA profiling on the second. It reuses unchanged fitting code and has150 seconds plus20 seconds termination grace, no automatic retry. Its maximum fits the remaining287.66 seconds of the original runtime-stage budget. Require both profile arrays to contain usable positive CUDA activity before naming a GPU bottleneck. Profiler-instrumented times do not replace unprofiled throughput.

After attribution, either qualify a minimal existing-library optimization inside the remaining authorized envelope, or mark the executable schedule budget-incomplete and resolve the resource/design constraint prospectively. Keep the8192 answer cap, all1500 fit and256 selection questions, ranks4/8, sites17/23 and both policies. Resident parallel workers may shorten wall time only after complete per-stage accounting is affordable. No budget is silently increased, no comparator omitted and no old closure reopened.

## Provenance and unresolved gates

### Completed profile and failed compiler-policy comparison

The profile described above completed in 87.984675437 container seconds. Its four steps and two CUDA profiles pass integrity checks; the independent critic verified 148 assertions. The forward kernel `triton_tem_fused_0` accounts for one attributed 9.884532924-second duration across 36 launches in the instrumented maximum-length step. All seven retained generated sources map that name to IEEE FP32 FlexAttention. The export omits event-device types and includes overlapping annotations, so durations must not be summed into a percentage of GPU or wall time. Dynamic specialization is a plausible explanation for the observed cost cliff, not a proved exclusive cause.

The separately frozen static-compiler diagnostic completed all 18 optimizer steps in 81.156233799 container seconds, but failed its final trajectory comparison: eight step pairs pass, ten gradient norms fail, and one loss fails. All sources, targets and the actual compiled-callable identity match. The optimizer receipt was written before this final check and cannot override exit 1. Even the first freshly initialized full/median step differs in gradient norm, so accumulated updates alone do not explain the entire discrepancy. Independent reconstruction checked 261 assertions without an audit discrepancy. The failed packet is retained at `artifacts/engineering/irc_static_fit_v1/`.

The static process's next-two-step mean times were approximately 0.148/0.452/2.842 seconds for prompt median/p95/maximum and 0.146/0.453/2.836 seconds for full application. These are sequential measurements, not independent repetitions or an admitted speedup. The failed comparator does not determine which engine is closer to mathematical attention.

Known measured runtime-stage use is now 0.867077294655 H100-hours, plus a separate 0.10-hour historical reserve: 118.521739242 seconds remain before the existing one-hour ceiling. `irc-static-oracle-config-v1.json` freezes one separate 100+10-second diagnostic using the existing FP64 output/QKV-VJP test at batch 2, 32 query heads, eight KV heads and shape history 384→896→2816→384. The singleton uses `training=False`, matching actual fitting, and must retain callable identity. Tolerances, IEEE precision and masks are unchanged. This prospective test concerns synthetic operator accuracy; it cannot rewrite the failed scalar trajectory comparison or automatically admit training, selection or a scientific result.

C77 records the actual-weight gradient/ownership observation; C78 records only deterministic padding counts. C67 remains unverified. The full16-question inference and synthetic VJP results remain C75/C76 at their own hashes and scopes. The remote CPU packaging failure, original runtime timeout, numerical failures and all earlier research losses remain preserved.

This update replaces pending-status summaries with retrieved evidence and adds the explicit cost gap; it adds no novelty, causal, behavioral or SOTA claim. Scientific fitting/selection, a useful full-generation action, response learning, development uncertainty, strongest-comparator frontier, three architecture families/three behavior constructs, independent full-paper review and current venue compliance remain open. Responsible-author approval and AI-use disclosure remain unverified.

## Separate oracle result and next integration

The separately frozen static FP64 test completed in 48.249238724 seconds with all four cases and 16 output/QKV-VJP comparisons passing. The maximum elementwise error divided by its unchanged tolerance is 0.7497476162. The 91,750,400 checked scalar instances are not independent trials: there is one seeded synthetic ensemble per shape and the repeated 384 case reproduces the first case's recorded error summaries. An independent critic checked 131 source, shape, terminal and accounting assertions, including exact reversal of the test's geometry-only parameterization to the original source hash. This supports C81, not the rejected C80 trajectory claim.

Runtime-stage measured use plus the historical reserve is now 0.980479860967 H100-hours, leaving 70.272500518 seconds. The next frozen native accumulation/checkpoint diagnostic uses at most 60 seconds plus five seconds termination grace, four unique fit questions, two logical optimizer updates and 16 repeated pair microbatches per update. Its raw full-module checkpoint must preserve parameter-chart state and last-valid-token logits exactly. It is not an eligible trained baseline.

Local accumulation tests exposed a test-fixture error: upstream LoReFT's compact save/load preserves the effective rotation but rebuilds the underlying orthogonal parameter chart. Losses matched while parameter-gradient norms differed. Copying raw unbound `nn.Module` state fixes the comparison without changing its tolerances; 36 scoped tests then pass with one CUDA skip, and the additional invalid-limit checks pass separately. Compact inference serialization is not an exact optimizer-resume contract. This finding does not explain the earlier static diagnostic, which initialized from the same seed without loading a checkpoint.

`irc-baseline-resource-amendment-v2.md` explicitly replaces only the unopened baseline-stage allowance with 48 H100-hours. All scientific conditions and the original engineering ceiling remain fixed. The new allowance is based on sensitivity arithmetic, not a guaranteed completion estimate; exact accumulated training, checkpoint generation and dispatch admission remain pending.
## Native accumulation completion and scientific baseline launch

The first accumulated probe failed before model loading because its snapshot omitted a hash-guarded author-source dependency. Its15.978087539 seconds remain charged. Explicit technical attempt2 added the missing immutable source, checked the real target loader on CPU, reused a verified read-only compiler cache and used a shorter48+5-second deadline. It completed in45.096359387 seconds: two logical updates,32 microbatches, four unique questions and64 repeated example instances. Global supervised-token instances total26,240. Both Adam step vectors and clipped norms pass the prospectively fixed contract; raw full-module state and last-valid logits match exactly after disk reload. This does not save or validate optimizer/scheduler/RNG resume.

`irc-accumulation-result-audit-v1.json` (SHA2563cacf9aa216b2a0c8ae9cd30c0bca6c2f0b60e255958db7e7f6ae3f35acdf1dc) includes176 independent checks and two append-only corrections to copied provenance text: attempt2 happened after the old static FAIL, and callable identity was checked after two updates, not18. Numerical thresholds and the frozen config are unchanged. Original engineering remainder is9.198053592 seconds including the separate historical reserve. Do not dispatch another native diagnostic or move these costs into the scientific baseline stage.

The complete baseline reuses the unchanged qualified fit-step body and existing sampler/evaluator, adding only epoch scheduling and generation-mask orchestration. The final config SHA256 is207e37dd9a7bf977378db65e41db53177451ee36a037e8d347beaa99c34a9acf. Local414 tests pass with one CUDA skip; exact native-runtime CPU preflight verifies52 sources, all targets,12 epoch-order hashes and49 tests. An independent source critic found no blocking defect. New integration is CPU-qualified, not a claim that every full-recipe GPU shape/rank/site has already passed.

All eight workers started22:40:18–19 UTC on7 September. Each handles its32-question zero share, one12-epoch candidate fit and all256 candidate selection rows. Three finite padding buckets entail11,131,392 padded token instances per candidate fit,45.53% more than within-logical-batch pairing padded to multiples of128. A sensitivity using static measured steps and prior unmodified rollouts gives33.30 H100-hours overall; it is not an upper bound, a measured full run or an admitted speedup. Each process has21540 seconds plus10 seconds kill grace, below its21600-second container envelope; total ceiling48 H100-hours remains. Outcomes, utility and the stronger-comparator/response/development/full-paper gates are pending.
