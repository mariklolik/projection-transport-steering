# Static / isolated-eager fit bridge v9

Frozen before any v9 GPU execution, 2026-09-08. Research-design increment only.
Goal remains measured SOTA steering across the frozen architecture/construct benchmark, not infrastructure qualification.
At entry all37 hashes in v8 closure c36531a200a71ad967751bc3fe7d4a9919551cf93a79d9f1a93d9ade28fae2dd matched.
Evidence is EVIDENCE_AVAILABLE / EVIDENCE_THIN; empirical S1–S9=[3,2,0,3,0,0,0,4,3],15/36, presentation=[3,3,4].
The defect is SCIENTIFIC_WEAKNESS: useful action, strongest matched baselines, effect/uncertainty and generalization remain absent.

## Discovery and exact integration contract

The frozen v8 source already contains the isolated native-mask clone, unchanged FP32 attention wrapper,
the existing grouped-FP64 output/QKV helper, run_irc_fit_probe, reft_training.fit_step and upstream LoreftIntervention.
No new trainer, attention kernel, model copy, scientific recipe or production source edit is needed.
The numerical helper accepts geometry, backend_name and check_no_grad; reuse its exact code with process-local
globals bound to the registered eager mask/attention, as in v8. Keep the four original comparisons, B2 and original buckets.
The fit probe calls fit_step(model,action,batch,site,application,optimizer) and consumes its result dictionary.
Reuse the original accumulation instrumentation: repeat the same B2 batch16times inside that existing helper,
with its global target-token denominator, adapter-only ownership checks, AdamW update and clip.
The wrapper captures the final action/model/batch for the existing raw nn.Module state/logit roundtrip.
Pre-register the existing alias before calling probe.main(); its unchanged old-alias-only branch is not relied upon.
Preserve both compiler reset boundaries and restore the original singleton callable after each stage.
The later baseline runner requires a separate one-condition registration change plus focused TDD and fresh admission.

## Question, alternative and decision

Does the exact isolated-eager mask compose with static compiled attention at original fit geometry
and with the existing accumulated actual-weight update/raw-checkpoint contract?
C81 and C82 covered the old mask; C89 covered only the new-mask inference workload.
The alternative is a mask/compiler/gradient/checkpoint interaction that those separate gates cannot exclude.
Success permits baseline source/cost admission only. It does not restore the failed historical trajectory parity,
establish useful LoReFT, prove all-shape compatibility, or support efficiency/SOTA claims.
Any numerical, finite-gradient, ownership, clipping, identity, checkpoint, OOM or timeout failure rejects this composition;
all partials and compiler output stay included. No favorable subset, tolerance change, automatic GPU retry or old-cache rescue.

## Frozen stages

A: B2,H32/KV8,D128,BF16 leaves/outputs; IEEE FP32 operator and grouped-FP64 oracle.
Shape history384,896,2816,384; seed20260908; training=False; output and Q/K/V VJP checks each at rtol0.005,atol0.0005.
The original safe-zero masked-row convention and mask predicate equality remain required.
All4 cases/all16 comparisons must pass. Assertion failures retain all scheduled cases; other technical failures stop.

B, only after A passes: exact Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218,
4 pre-existing fit questions at original median/p95 groups; rank8/site17/full; seed20260907;
two updates,16 repeated B2microbatches each, lr0.0009, clip1 with original0.005 record tolerance.
No new data selection or target truncation. Expected lengths/target hashes, installed runtime and source hashes are checked.
Raw adapter keys/tensors and last-valid-token logits after reload must match exactly.
This tests2optimizer updates,32microbatches,64logical instances but only4unique questions,
not optimizer/scheduler/RNG resume, full12epoch training, maximal-length actual-weight fitting or held-out efficacy.

Both stages use new independent cold Triton/Inductor/CUDA caches, Python3.12 -S,
pinned image/runtime/RO v8 source/RO model, networknone, one idle FSK42 H100.
A child160s+10sgrace / inclusive ceiling200s; B child560s+10sgrace / inclusive ceiling600s.
Combined reservation800s comes only from2367.256645422remaining additional-engineering seconds.
No baseline transfer; baseline remains40.394625745277778H100h. The ceilings include startup/teardown reserve.
A stage failure stops dependent GPU work; independently frozen unused budget is not spent automatically.

## Statistics, provenance and continuation

Technical cells are deterministic checks, not independent statistical samples. No confidence interval or efficacy estimate applies.
The4fit questions are reused from C82, not a new validation population. Final-test boundaries remain untouched.
Raw output takes precedence over deterministic audit, machine receipt, human summary and notes.
Record config/argv/source hashes, native CPU input/registration check, independent source/argv critic,
actual GPU lifetime, every attempted container, terminal state, source/output hashes and compiler artifacts.
v8 inference and completed literature/benchmark gates are reused, not rerun.
No submission prose, skeleton or fair-strong claim is admitted by this increment.
If both stages pass, freeze only the needed baseline registration wiring and prospective inclusive cost scenario;
then run all8 fresh candidates and a fresh complete shared zero under the same engine.
No three-survivor reuse, mixed-engine contrast, cap change or metric substitution is permitted.

