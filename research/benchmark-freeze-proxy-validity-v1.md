# Teacher-forced proxy validity benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before generation or judge output.

Use only concept IDs `1,5,16,24,29,33,34,48,58,59,60,73`, their existing eight evaluation prompts, and the 14 conditions recorded in `artifacts/development/flow_step_support_v1_design_attempt1`. No construction prompt, neutral row, fresh concept, pilot, held-out concept, or 9B model may be generated.

Use the released Gemma-2-2B-IT FLAS checkpoint at layer 20, flow time 2, and the exact raw, equal-time, full, and `N=2` condition definitions already frozen for Flow-step support version 1. Decode greedily with maximum 128 new tokens. One deterministic output is produced per concept, prompt, and condition. Empty output or any missing/duplicate key fails the corresponding shard; no selective regeneration is allowed.

Judge every output once for Concept, Instruction, and Fluency using the frozen Qwen3-14B local proxy, upstream FLAS judge prompts, upstream rating parser, greedy decoding, maximum 160 new tokens, and batch size 8. The model configuration SHA-256 is `e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a`; its sorted top-level filename-and-size manifest SHA-256 is `78a536738d6391c113dc4a5c239644b4336c9db223bba7a576d07c7cb6a1055b`; the upstream judge source SHA-256 is `2a8997810b1930d817262e9c3c249299cdda8804ddab67d72057d1ebe7b62755`. Preserve raw completions. A missing parse remains missing and cannot be replaced or mapped silently to zero. Any condition-wise parse-rate difference or incomplete packet fails PV.1.

The analyzer is frozen before generation. It joins by exact concept, condition, and prompt indices; validates 1,344 generations, 4,032 completions, source hashes, model manifests, condition definitions, forward/generation counts, and serialization; then applies only PV.2--PV.5. Spearman ties use average ranks. Bootstrap resamples the 12 concepts with 10,000 draws and seed `20260907`. No Pearson substitution, pooled-row correlation, observed-condition selection, threshold change, favorable judge retry, or alternative generation can enter the decision.

Run one concept and one prompt across all 14 conditions as an engineering smoke. If valid, run six two-concept generation shards on six H100s, then six matching judge shards. Keep at most two CPU and interop threads per worker. Analyze only after all hashes and counts pass.
