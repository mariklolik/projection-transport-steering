# Real-question runtime and baseline-readiness audit

2026-09-07. Engineering results on the 16 fixed fit questions; no new steering-policy comparison. This goal turn is **progress**, not completion or a repeated external blocker: it added a tested executable, immutable real outputs, an independently audited timeout, a measured operator bottleneck and primary-source corrections to baseline design.

## Decision

The real-question runtime gate is **FAIL/incomplete**. All three 512-step profiles passed their frozen within-condition replay, RNG and cache checks. The separate 8,192-token workload reached its 900-second timeout after saving only the first eight questions. It has no final receipt. No adapter fitting, response collection, development or sealed test is authorized by this packet. The model's wrong or capped answers are retained outcomes; the missing second batch is a distinct technical incompleteness.

The prospective config is `irc-real-runtime-config-v1.json`, SHA-256 `b1d9fee872e2cdf35844fb7453c3cc35104acd494212bf39b46d29be63292cfb`. Its protocol, exact source/model/runtime identities, 16 ordered groups and exclusive-output contract were frozen before GPU generation. Four detached, deadline-bounded H100 workers ran concurrently on `avi-gn-fsk42`; a dropped observation connection could not relaunch them. The original native Torch 2.13 replay failure remains rejected. These runs used the isolated Torch 2.9.1/Transformers 5.16.1/Pyvene 0.1.8 environment, not an installed copy of the current local lockfile.

## Complete accounting

| Workload | Saved / declared questions | Emitted reference tokens | Reference seconds | Reference tokens/s | All container seconds |
|---|---:|---:|---:|---:|---:|
| 512 steps, batch 1 | 16 / 16 | 8,192 | 186.711 | 43.88 | 601.785 |
| 512 steps, batch 4 | 16 / 16 | 8,192 | 52.824 | 155.08 | 183.695 |
| 512 steps, batch 8 | 16 / 16 | 8,192 | 28.167 | 290.84 | 106.543 |
| 8,192 steps, batch 8, incomplete | 8 / 16 | 42,950 | 471.081 for saved batch only | 91.17 for saved batch only | 901.337 including failed work |

The table describes one execution per condition on 16 reused fit questions, not independent repeated-run confidence intervals. Reference timing excludes checks and grading; container time includes imports, loading, checks, grading and unsuccessful work. The incomplete full-budget row is not a complete-workload throughput estimate. Its 42,950 emitted tokens must not be replaced with 65,536 padded batch slots. The four container intervals total 0.498156 H100-hours; millisecond-rounded values are stored separately from nanosecond timestamp strings. This is measured allocated process time, not integrated GPU utilization. Earlier engineering and the later operator diagnostic are additional costs.

The short profiles each have 16 capped outputs, zero EOS and zero primary correctness. That is expected truncation accounting for this profiling workload, not a source-admission failure or a scientific claim about the shorter budget. All 48 short-profile rows are at risk at cut 256 and receive 64 action tokens; all 22 batch check records pass. Peak allocated memory is 17.50/20.79/25.15 GB for batch 1/4/8, including the replay/branch workload.

The saved full-budget batch has six EOS-terminated correct answers and two cap-8192 failures. One capped answer is extractable but remains incorrect under the unchanged termination rule. Primary and sensitivity scores agree on these eight rows, without evaluator exceptions. The other eight outcomes are unobserved, not imputed failures or excluded from a claimed complete-panel accuracy. Even the arithmetic possible range for all 16 is 6/16 to 14/16, not an accuracy estimate or confidence interval. Subject/difficulty counts are retained in the machine audit; such small incomplete strata support no generalization claim.

There are 56 saved scored rows out of 64 declared, with only 16 unique questions. All 49 transferred output files match remote SHA-256 values. An independent critic recomputed all 56 raw-reference/scored joins, native prompt/token hashes and decoded completions, checked the source/config inventories, and confirmed the fail-closed interpretation. Scorers were not rerun: exactly-once evidence consists of the pinned call path, tested exception handling and absence of duplicate output identities, not a separate runtime trace of every scorer call.

Cross-batch token equality is not supported: only 2/16, 1/16 and 1/16 full short continuations match across batch 1–4, 1–8 and 4–8. All prompt and seed identities match, and within-condition replay is exact. The first 512 tokens of the eight saved full-budget continuations match their batch-8 profiles. This supports only the declared execution contracts, not cross-engine identity.

## What the slowdown diagnosis establishes

The installed Transformers SDPA wrapper, SHA-256 `53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f`, disables native GQA when a mask is present and expands both KV tensors. A separate prospective synthetic operator diagnostic tested query length 1, batch 8, 32 query heads, 8 KV heads, head dimension 128, bfloat16, key lengths 512/2,048/8,192 and no-mask/all-true/left-padding masks. No model weights, task answers or candidate adapter were used.

At length 8,192, the measured mean over 32 timed calls was 0.0989 ms without a mask and 1.5791 ms with left padding; the latter added 1,073,938,432 bytes of peak allocation above its resident inputs. The profiler identifies two KV copies and the efficient-attention path for masks, versus the flash-attention path without a mask. At 512/2,048 the masked calls take 0.1321/0.4322 ms. This establishes a length-dependent operator cost, not a measured 16-fold end-to-end model speedup or proof that it explains every second of the timeout. Absorbed/padded work, cache updates, model projections and Python/sampling overhead still matter.

Attempt 1 stopped before timing because the all-true/no-mask comparison exceeded the declared rtol 0.005 / atol 0.0005 in one of 32,768 bfloat16 outputs. Attempt 2 kept the same tolerances, inputs and timing schedule, recorded that failure and completed all nine cost cells. All nine zero-query masked-mean checks passed. The numerical difference is retained, not relabeled exact parity. The two diagnostic containers ran about 7.19 and 9.28 seconds, adding about 0.00457 H100-hours. Removing a real nontrivial mask would change the model and is not an admissible fix.

## Baseline correction and next step

The [baseline design audit](irc-fixed-adapter-design-v1.md) rechecks the ReFT paper and pinned author example. Their arithmetic recipe uses 12 epochs, all layers and first/last-seven prompt positions; the one-epoch CLI default and a single-layer adaptation are not that experiment. The source-only target audit verifies all 1,500 fit solutions and a maximum formatted length of 2,690 tokens. A finite single-layer action search can supply an IRC action family, but cannot stand in for the strongest resource-envelope ReFT/LoRA/router comparisons.

Next qualify an existing **mask-preserving GQA backend** in the admitted runtime before changing production inference. The installed Transformers FlexAttention path is an available callable candidate, not a verified fix; include compilation, mask construction, padding/EOS and cache/replay behavior in its cost and conformance tests. Keep the old timeout and outputs immutable. Any new engine requires a prospective versioned receipt and the same 16-question/full-budget contract; do not change the 8,192-token cap, choose easier questions, remove masks or use partial correctness to tune it. The one-H100-hour engineering ceiling and eight-H100-hour baseline ceiling remain in force. Only after runtime admission should an actual-weight fit-step cost and fully specified fit/selection schedule open baseline fitting.

## Verification and claim boundary

The final local suite passes 356 tests with 25 existing/upstream warnings; critical Ruff checks pass. All 15 runtime tests also pass in the exact remote isolated CPU environment. The retained first remote test failure came from a child process losing `-S`; the test now preserves its parent's isolation flag. Independent fault-injection tests verify that a completed reference is saved before later checks/scoring, and absorbing rows do not claim an executed intervention. Old source functions are AST-identical under `irc-real-runtime-code-invariants-v1.json`; additive helpers do not rewrite historical graders or allocation logic.

The audit and planning prose are `DRAFT_FROM_EVIDENCE`, not submission prose. C69/C70 keep their prior narrow engineering/data scope; C67, useful adapter, learned response controller, full-rollout effect, strong-comparator, architecture/behavior breadth, fair-strong and ICLR-readiness gates remain unpassed. No v20, observer-v1 or other closed route is reopened. Raw outputs and logs remain both recoverable in this evidence packet and, for generation outputs, in the remote bind-mounted output directory; only the six stopped task-owned containers were removed after capture.
