# Complete fresh fixed-adapter baseline v3

Prospectively frozen after v8 inference and v9 static/eager composition, before any v3 GPU execution.
This supersedes only v2 engine/dispatch/resource allocation, never its scientific recipe or adverse evidence.
All v2 partials stay excluded from this complete same-engine attempt; none become selected survivors.
User-authorized autonomous FSK research is the authority; the assistant-defined ceilings are not claimed as user-specified.

## Discovery and minimal implementation

Reuse run_irc_baseline.main, prepare_inputs, train_adapter, evaluate, reft_training.fit_step,
epoch_batches, generation_intervention, native grouped-FP32 attention and upstream LoreftIntervention.
The only production source change extends the existing backend-registration condition to
fp32_prefill_flex_eager before model loading. Existing static singleton/reset/restore,
all data/scoring/seed policies, saved raw checkpoint and full CLI contracts remain unchanged.
The focused new registration test first failed exactly for the eager alias; the minimal change passed
15 relevant local tests (21 existing upstream syntax/deprecation warnings), including complete tiny CPU CLI paths.
Exact native CPU input/source/version and relevant runner tests, plus independent source/argv/cost review,
are required before dispatch. Matching previous inference, source and mathematical gates are reused, not rerun.

## Unchanged selection experiment

Eight fresh independent candidates: rank4/8 x site17/23 x prompt/full.
All1500 immutable fit questions, native public solutions, seed20260907,12epochs,
effective batch32,microbatch2; exact original logical membership then within-group length sorting;
buckets384/896/2816; no truncation. All564 updates,57 linear-warmup updates,
AdamWlr0.0009,betas0.9/0.999,eps1e-8,weightdecay0,dropout0,clip1.
Base frozen/eval; raw final-epoch adapter only, unitstrength1. No early stopping or intermediate selection.
Keep both process-local compiler resets, static fit callable identity and original default dynamic inference callable.

Each worker first evaluates its original disjoint32-question zero share, fits, then evaluates
its final adapter on all256 selection questions in original batch8 order.
Prompt first/last7 valid positions; full additionally applies to cached generation positions.
Temperature0.6,top_p0.95,top_k20,original per-question seed0,8192-token cap and original graders.
Invalid/unparsed/truncated answers remain zero in the original primary denominator.
No cap, metric, source selection, label, loss target or padding policy change.
Technical/evaluator failure preserves all raw chunks and makes that candidate incomplete, never defeated.

Select full and prompt winners separately by original exact primary correctness,
ties lower rank then lower site. Full must strictly beat the fresh complete shared zero
before response collection can open. All8 and allzero256 rows are required for that admission.
This is exploratory initialization, not a confirmatory efficacy result or SOTA comparison.
The benchmark's strongest comparators, protected outcomes, uncertainty,3architectures/3constructs remain required.
Question/cluster is the later paired unit; no token/microbatch independence or exploratory winner confidence claim.

## Resource decision and conditional affordability

FSK42 only, at most8currently idle H100s, one candidate/GPU and one base load/worker.
Fresh source snapshot; independent cold compiler/CUDA caches per worker; no old adapter or zero reuse.
V2 consumed7.605374254722222of48H100h, leaving40.394625745277778.
Reserve40H100h for this attempt: each child17940s+10sgrace within18000sinclusive lifetime.
This leaves1420.652683seconds outside the eight initial ceilings. No engineering-budget transfer or automatic retry.

Planning sensitivity, not a bound: use the source-only schedule's larger fit scenario
(3101.553924523294seconds/candidate) and the slower of v8's two complete capped batches
(331.8252535201609seconds per batch8). Each worker has4zero and32adapter batches.
Add600seconds/worker for loading, compile/reset, checkpoint, grading and other unmodeled overhead.
The latter is a prospective allowance, not an observed duration.
Adapter-generation multipliers1.00/1.15/1.30 give respectively
4.346462/4.788896/5.231329worker-hours
(34.771696/38.311165/41.850634totalH100h).
The1.15scenario fits;1.30does not. The actual break-even adapter multiplier is
1.221572.

Choosing the bounded complete attempt is an explicit engineering judgment, not proof that all8 will finish.
The fit estimate uses old-mask synthetic/timing evidence, not guaranteed new-engine throughput;
v9 proves bounded composition, not a full-recipe speed estimate.
Adapter overhead, longer contexts, eight-way contention, compile history and dataset mix remain uncertain.
The strict deadline must close budget-incomplete output honestly if these assumptions fail.
Do not spend the leftover engineering pool on further unchanged readiness loops.

## Admission and reporting

Freeze config/source/model/data/runtime/argv hashes; require the v9 outer checkpoint receipt and exit0,
not merely probe.pass. Native CPU validates all1500 targets, selection disjointness/zero partition,
all12 epoch hashes and unchanged bucket histogram; no GPU is used for this.
Independent critic reviews the minimal diff, exact argv, complete-candidate policy and conditional-cost interpretation.
Launch only from fresh occupancy checks on exact GPU UUIDs; do not affect unrelated processes.
Record actual container ids/lifetimes, every failure, complete/partial row denominators and checkpoint identities.
Validate completed chunks on CPU while GPU jobs continue, without repeatedly scoring incomplete files.
No intermediate outcome may alter candidate, schedule, cap, primary endpoint or stopping rule.
The broad goal stays active. Paper prose and SOTA claims remain blocked until decisive scientific evidence exists.

