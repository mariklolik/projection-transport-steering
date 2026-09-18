# Source probe v2: explicit engineering recovery

Frozen 2026-09-08 after v1 terminated, before any source generation. Inherit every data, sampling, stage, measurement and scientific-boundary rule in `source-probe-protocol.md`; that immutable original and packet remain archived. V2 permits exactly one additional GPU launch, not an automatic restart of v1.

## Observed failure and narrow change

V1 container `215a0e875a0c44056dfda6af245c52f1720b66482f2da4102947c04ffcd7acfa` exited1, no OOM, zero stages, no model generation. Native StartedAt11:28:49.399808340Z and FinishedAt11:29:23.919344613Z give34.519536273s =0.009588760075833 H100h. FlashInfer is imported only when the lazy Engine entrypoint loads; it ignores XDG_CACHE_HOME and defaults FLASHINFER_WORKSPACE_BASE to Path.home(), yielding an unwritable /.cache for UID11027. This was missed by the earlier ServerArgs-only CPU import. Set FLASHINFER_WORKSPACE_BASE=/tmp/action-source-probe-flashinfer, retain the other task-local writable caches, and CPU-import the actual Engine module before dispatch. No package/image, privilege, model, data, sample count, seed, cap, precision or attention change.

## New-code contract corrections

The independent critic identified missing token upper-bound validation and cleanup exceptions suppressing the final receipt. Add the hashed model's vocab_size to the packet and reject token IDs outside it. Cleanup must attempt all releases, preserve the original error, record cleanup errors and always write elapsed/final status after caught cleanup exceptions. A repeat-invariance failure must persist its receipt and produce a nonzero process exit. Source inspection confirms EOS precedes the cap in this pinned SGLang's update_finish_state; EOS at exactly8192 is therefore still an EOS stop. No EOS/cap rule is relaxed.

V1 source code is preserved in `tmp/action-source-probe-v1.tar`, the immutable remote v1 input mount and the original packet's source hashes. V2 source/test/packet/launch hashes are separate. The only newly admitted information is engineering failure and its correction; no outcome-dependent scientific selection occurred.

## Admission and cost

Require local new-contract tests and actual-image CPU unit tests plus lazy Engine import with the same environment before v2 CUDA. Reuse the exact verified image and all model file hashes; retain read-only mounts, one freshly idle H100, network isolation, external1740s timeout with30s kill grace and1800s inclusive ceiling. Prior actual cost now23.063021006186943/48 H100h. The v2 worst-case0.5 H100h fits without resource reallocation; both attempts are charged. No further retry is admitted by this protocol.

All eight fit questions,32 unique source requests,38 full-cap completions plus one16-token warmup and all repetition rules remain byte-for-byte identical in the two request/stage/sampling/engine objects. The empirical SOTA goal, useful-action requirement, full source/backward/reference/four-fit/evaluation affordability and protected-utility/three-family/three-construct gates remain unpassed.
