# CacheBack GCAD compatibility audit, version 1

Date: 2026-09-06. Decision: GCAD is not a compatible comparator for the frozen GPT-2 morphology pilot. Its omission forbids a frontier-superiority claim but does not block the CacheBack development comparison.

## Audited method

The audit uses arXiv:2605.10664v1 and public code commit `6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de` from `NihiI-obstat/Gated-Cropped-Attention-Delta`.

GCAD is not an alternative direction inside the same single-site residual actuator. Equations 7 and 8 extract a separate vector at every steered layer by retaining only system-prompt source-token contributions to attention and differencing positive and negative prompt-response sets. Equations 9 and 10 additionally require a per-layer mean post-RoPE system-prompt key and a response-token demand mean. Equation 11 injects the gated vector into each selected attention output before the MLP at every decoding token.

The released implementation confirms this contract. `core/generate_vec.py` locates an explicit chat system boundary, exposes full attention weights, crops source positions to that boundary, and averages the cropped output over response positions. `scripts/generate_vec_prompt_attn_k.py` separately fits per-layer positive-prompt keys and demand centers. `core/gated_steerer.py` applies distinct vectors and query-key gates over a configured set of Qwen/Llama attention layers. The released main configuration uses eleven layers, not a one-site action.

## Frozen-packet mismatch

The CacheBack packet contains three token-level morphology mappings and raw C4 source states. Each mapping provides base and target token IDs and paired token strings. Each state provides token IDs, a source-document hash, a token position, and a sealed stage. It contains none of the following GCAD-defining inputs:

- positive and negative system-prompt pairs;
- a system-token span inside each source sequence;
- generated positive and negative response sets;
- per-layer cropped attention-delta vectors;
- per-layer positive-prompt keys and demand centers;
- a multi-layer attention intervention budget.

GPT-2 also does not satisfy the released implementation's Qwen/Llama module contract without a substantive port. More importantly, a port alone cannot supply the missing construction data or make an eleven-layer, per-token attention actuator equivalent to the frozen one-site residual action.

## Rejected substitutions

Using the already fitted CAA vector at the attention output would omit prompt cropping, the positive-prompt key, and the token gate, so it would not be GCAD. Inventing morphology system prompts and response sets now would change the construction distribution after the freeze. Treating the raw C4 prefix as a system prompt would create a method not defined or validated by GCAD and would give different methods different fit information. Matching only total vector norm or runtime would not repair the actuator and information-budget mismatch.

## Consequence

CAA/ActAdd remains the required construction-matched direction baseline. Exact FishBack H0, Euclidean, CAA, the future horizons, cache-path controls, and top-5000 remain in scope. GCAD is recorded as an important mechanism-adjacent method and a limitation, but no GCAD result will be fabricated for this pilot. Therefore CacheBack cannot claim superiority over attention-side steering, GCAD, or the steering frontier from this benchmark.
