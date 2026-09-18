# IRC v3 runtime checkpoint

Current observation update: the earlier pending checkpoint below is retained as history. The complete full packet has now been recovered and independently audited in `irc-mixed-full-result-audit-v3.json` (SHA256 `62da42faf8871d3767a2db9c5e7cedec41653af67f5be65d1be34ab437d9f7af`):16/16 original questions,8 correct EOS/8 cap failures, no evaluator errors, exit0/noOOM,660.672601913 container seconds. No duplicate job or rescoring occurred. The unchanged synthetic CUDA backward test also passes; `irc-native-flex-backward-result-v1.json` records its31.403719810-second interval. Actual-weight fit-cost qualification is now the active dependency; scientific fitting, C67 and SOTA remain unpassed.

2026-09-07 16:22 UTC. DRAFT_FROM_EVIDENCE. The complete short profile is admitted at its exact scope; full-runtime completion is not yet verified because SSH access was lost after partial worker progress.

## Verified

The named `fp32_prefill_flex` backend completed all 16 fixed questions at batch8/512. Identity and split/RNG replay, parent-cache isolation and output accounting pass. All16 rows were at risk at the cut, and each received 64 nonzero action tokens; four rows changed token outputs. An independent critic reconstructed 314 integrity checks without violations, including source/config hashes, question seeds, native prompts, raw/scored joins and replay tokens. This is a count of audit assertions, not independent experimental replications.

All16 references exhausted the512-token profile cap without EOS. They do not establish answer quality or long-rollout capability. Reference generation emitted 8192 tokens in 71.599771 seconds (114.4138 tokens/s), while the whole container took 174.175383420 seconds including checks, imports and compilation (47.0331 reference tokens/container-second). Peak allocated model memory in the receipt is 25,060,251,136 bytes. One run supplies no stable speed ordering or timing interval.

The result source is `irc-mixed-profile-result-audit-v3.json`, SHA256 `6b37eac89f5a5f40f83cc10a132151baef96cf044f95aa651ac6e22efa4ae745`; the worker receipt is `b045e178e36c8930f01460a33a6f652fa3f768b66ed0d896f988477e61a4b75a`. The earlier v1 SDPA timeout and v2 BF16-prefill numerical failure remain immutable.

## Not yet verified

The original 8192-token,16-question condition was launched only after the complete profile checks. Worker `80711276d2b052a2aadb8cd75c2feb69f197ba3ecfe56285d3b3248891f735cf` last reported 8 completed rows,44160 reference tokens and325.963840 reference seconds. No full-condition JSON packet has been retrieved locally; these are progress-log observations, not an audited partial score or full-panel estimate. The container has its original900-second deadline plus20-second kill grace, but terminal state still needs direct observation.

The separate native-backward test is prepared and source-reviewed. Its SSH dispatch returned255 without a container ID; whether a worker exists under `pts-irc-native-flex-backward-v1` must be checked before any retry. A local CUDA skip is not a backward pass. No actual GPU0 overlap with the full job is asserted. The unchanged runtime envelope conservatively reserves this attempt as well as the entire full-condition deadline.

## Access and next action

Fresh target and jump-host SSH both time out before a banner. `dc-fixed` reports PASSWORD_WAIT, and the bastion route uses en0. The existing reconnect helper generated a redacted code and attempted login, but timed out; Tunnelblick still presents its login dialog. Happ and SSH configuration were not changed. User authentication must be completed in Tunnelblick, not in chat.

After access returns, inspect the exact full container and named backward worker, preserve terminal states and logs, transfer the existing output directory and verify remote/local hashes. Never duplicate a possibly completed job. Audit all 16 full rows and the final receipt before fitting; if incomplete, retain the failure and reconcile the remaining budget before any new engineering decision.

## Research and provenance boundary

Seven diagnostic/source/runtime skills route this checkpoint through claim, results, strengthening, setup, prose and local checks. The current local suite has 363 passes, one CUDA skip and25 warnings; the frozen profile snapshot passed 22 exact remote CPU tests before GPU execution. Later test-only additions do not mutate that snapshot.

The fixed-adapter source audit now identifies reusable upstream loss/collation code and the missing position, EOS, truncation, accumulated-token-weighting and native-backward contracts. No adapter or controller has been fitted. C67, meaningful efficacy, strongest comparators, uncertainty, multi-model/multi-behavior breadth and fair-strong evidence remain unpassed. Paper section writing and ICLR submission compliance remain downstream. No old route, scoring rule or generation cap is reopened.
