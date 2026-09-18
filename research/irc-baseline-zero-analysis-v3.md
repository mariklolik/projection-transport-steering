# Fresh baseline-v3: complete zero-reference audit

## Scope and decision

The complete shared zero-reference contains the original 256 selection questions, generated afresh by Qwen3-8B on eight disjoint 32-question shares. It is an exploratory reference for selecting the original eight fixed adapters, not a sealed-test estimate, intervention result or SOTA comparison. One rollout per question does not establish across-seed stability. Fit, response and protected-test allocations remain separate; no protected-test outcomes were opened.

The existing native CPU auditor verified every raw/scored identity, prompt/token hash, decoded completion, per-question seed, fixed sensitivity-support membership and EOS/cap boundary, then reproduced both stored graders once. All 256 rows passed. The condition input manifests reconstruct expected source IDs and captured hashes; they are **not** producer terminal receipts. Final worker status, checkpoint/base integrity, complete eight-candidate selection and budget compliance remain open.

Evidence: [native result](../artifacts/selection/irc_baseline_v3/zero-audit-v1/result.json), SHA `14e1bc01402e39ff25a90fd160e0cd2606add34ff38a1cce992cc203ca8000ac`; [independent review](irc-baseline-zero-independent-review-v3.json); [closure and trace](irc-baseline-zero-closure-v3.json). Config `61529de3e7cf2e865d147ca11e54388590bb50e12d8702c388adb2ed7702f7be` and the [original analysis contract](irc-baseline-analysis-contract-v2.md) are unchanged.

## What the zero condition establishes

Primary correctness is **198/256 (77.34375%)**. There are 213 EOS completions and 43 cap-exhausted outputs. The mutually exclusive primary-outcome partition is:

| Outcome under the frozen primary rule | Questions |
|---|---:|
| Correct | 198 |
| Cap exhausted at 8,192 tokens | 43 |
| EOS, author-extracted answer, primary false | 15 |
| EOS, author extraction failed | 0 |

Thus cap exhaustion accounts for 43/58 (74.14%) observed primary failures. This is a statement about the frozen terminal-answer endpoint, not proof that those 43 questions were intrinsically unsolved or that merely changing the stop rule would repair them.

All 32 malformed extractions occur within the 43 cap cases. The remaining 11 cap cases have a nonempty author-extracted answer but still score zero because EOS was not reached. Malformed and cap counts overlap; they must not be added. No cap is excluded or relabeled, and the 8,192-token limit stays fixed.

## Where adverse outcomes concentrate

These are descriptive source-difficulty strata of the same reference panel, not new independent experiments or causal difficulty effects.

| Source difficulty | Questions | Primary correct | Cap exhausted | EOS primary false |
|---|---:|---:|---:|---:|
| Level 1 | 16 | 16 | 0 | 0 |
| Level 2 | 53 | 50 | 1 | 2 |
| Level 3 | 57 | 51 | 4 | 2 |
| Level 4 | 55 | 36 | 15 | 4 |
| Level 5 | 75 | 45 | 23 | 7 |

Levels 4–5 contain 130/256 questions and 38/43 cap failures. Intermediate Algebra contributes 18/43 caps among its 57 questions (36 primary correct); this is not evidence that the same ordering holds on another allocation. Complete subject and original prompt-length strata remain in the native result. The >=256 prompt-token bin has only ten questions, so a large-looking rate there is not a robust generalization.

## Grader disagreement is bidirectional

The fixed supported subset has 255 questions: primary 198/255 versus Math-Verify 203/255. Thirteen paired disagreements comprise **nine sensitivity-only** and **four primary-only** successes, not thirteen newly recovered correct answers.

Eight sensitivity-only rows show visible algebraic or numeric-format alternatives: radical/fraction notation, term ordering, thousands separators, sign placement, matrix fraction notation and an equivalent fraction/square form. The ninth, cluster `b17f7c806e6305742d01a8336d475a04aa1c00b32f6f23bcb31dab8c93b2b9d6`, is context-sensitive: the source asks for eleven in base two, while the extracted answer omits the gold's base subscript. This does not establish general base-aware grader validity. Four primary-only cases have unit suffixes in the gold but not the prediction.

The sole unsupported question, `7443dcbc81608b149ddc9a6e94251bfbdafb942f22d969141bc4170815152588`, is an EOS primary-false Prealgebra/Level-4 output with extracted `17:00`, not a cap or exclusion. Therefore the 15 EOS primary-false cases partition into nine sensitivity-true, five supported-both-false and one sensitivity-unsupported.

These are format/context-linked measurement discrepancies, not thirteen proven grader bugs or nine verified reasoning successes. Stored primary and sensitivity scores remain unchanged. The supported-subset analysis cannot replace the full primary denominator or choose a different winner.

## Cost and execution boundary

The zero reference emitted 1,143,747 tokens. The 43 capped rows contribute 352,256 (30.80%) of them. Summed generation-batch time is 9,846.580573897809 seconds (2.7351612705271693 H100 component-hours); grading is 2.073363985866308 seconds and check accounting 0.00129004567861557 seconds. These component timers are nested inside the still-running GPU workers. They are neither complete worker cost nor per-question latency, and must not be added again to terminal container charges.

CPU replay took 14.171910244971514 audit seconds and 15.35336460545659 controller seconds, with no GPU requested or model weights loaded. The unchanged analysis helpers reuse their matching 48-native-test admission; no generic test suite or GPU qualification was repeated.

The complete zero archive is 19,107,840 bytes, SHA `4e2e461193f328eaf92ac2e5fc09f878378deaa457cd07250386fe69af37b01a`. Its canonical inventory, SHA `8410ba1c3180598f9da40905675d0d07f44d4b8459c614a2783d1555bd04b8a2`, binds 90 files, including all 64 raw/scored output files and closed-chunk/live-container capture evidence. No old-v2 output, old adapter or rerun is included.

At 06:11:49 UTC on 8 September all eight original workers remain running, at updates 221–282 of 564, without OOM or restart. Instantaneous training utilization was 95–100%; this is not average utilization, peak-memory evidence or measured speedup.

## Unchanged next decision

Complete every original fit and every 256-question candidate condition before selecting full and prompt winners with the frozen primary-count/rank/site rule. The best full candidate must exceed 198, so the smallest possible positive entry is 199/256, a 0.390625-point gain. This is only the already-frozen exploratory initialization criterion, **not** an adequate SOTA effect threshold, a significance test or automatic response-launch authority.

The existing paired analysis must expose rescue/harm counts, changes among zero-EOS versus zero-cap questions, supported-primary versus sensitivity effects, source strata and token costs for **all eight** candidates. Candidate output length is post-treatment and cannot establish a reasoning mechanism. Its 10,000-resample question-paired BCa intervals are exploratory, neither simultaneous nor winner-corrected; they do not measure fit-seed variation.

Reuse this zero-component receipt only when all source/config/condition hashes and final producer joins match. Do not rescore unchanged zero files merely to produce another green receipt. No candidate, cap, metric, schedule, model family or scientific success criterion changes. Scientific strength remains 15/36, with S5 and S7 at zero. The original three-family/three-construct, strong-comparator and protected-evaluation goals remain required.
