# Residual-flow research plan, version 1

Freeze date: 2026-09-07. Status: active independent successor after SemanticOutcomeBack version 1 closure.

## Decision sequence

1. Preserve version 20, InvariantBack, CacheBack, and SemanticOutcomeBack closures and all invalid-attempt lineage.
2. Correct the literature corpus for UniSteer and freeze the residual-flow nearest-work, method, benchmark, claims, source hashes, and stopping rule.
3. Extend the existing robust QP to vector right-hand sides with deterministic scalar-equivalence, infeasibility, and KKT tests.
4. Build only the FLAS teacher-forcing adapter and one-concept design smoke, reusing the released generator, existing matched continuations, score functions, metric constructor, action hook, and receipt helpers.
5. If the smoke passes, run six two-concept design shards on six idle H100s. Each process loads the base model and FLAS checkpoint once and batches method-condition evaluation. Use at most two CPU threads per worker.
6. Run the already frozen concept-level analysis. Close immediately if no target quantile passes; do not spend generation or judge calls.
7. If the design passes, freeze the smallest passing quantile, materialize and hash only the 24 fresh successor-development concepts, then run tuning generations and exact or explicitly bounded judge evaluation.
8. Open held-in pilot, held-out 2B, 9B, and cross-family work only through their preceding gates. Re-run claim, A-star, strengthening, results, review, and compliance audits before paper prose.

## GPU efficiency

The design reuses version-1 matched continuations, avoiding 192 new autoregressive constructions. The trajectory-local metric is computed once per concept and reused across all three target quantiles and controls. FLAS finite changes and covectors are computed once per witness; no model is reloaded between the two concepts in a shard. CPU packet validation and the frozen analyzer run locally while GPU shards execute.

## Current research gates

The claim-evidence gate is `BLOCKED`: the surviving conjunction is not yet demonstrated and all novelty components have strong prior owners. The A-star contribution gate is `BLOCKED`: a corrected released flow is only a hypothesis until it beats unchanged FLAS and matched controls on fresh behavior. The strengthening gate is `PASS` for the prospective increment because it responds directly to the version-1 failure, changes the action class rather than retuning the closed objective, uses the strongest released flow as its base, and reserves fresh concepts for the next decision.
