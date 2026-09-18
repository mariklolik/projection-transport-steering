# Claim-relative steering research plan, version 1

Freeze date: 2026-09-06. Status: active research plan after the fail-closed closure of version 20. The `research-paper-writing` router classifies the project as `EVIDENCE_AVAILABLE` but still below `fair-strong`; prose assembly and combined method branding remain closed.

## Scientific decision

Versions 18-20 changed how the FishBack linear system was approximated while preserving the same next-token objective. That line is closed. The next experiments change the protected functional to match the behavioral claim and test two independent hypotheses:

1. InvariantBack asks whether an action constrained to move every construction encoding toward the same semantic endpoint generalizes to unseen encodings better than a mean covector or nuisance removal.
2. CacheBack asks whether the output-distribution metric accumulated through autoregressive time differs materially from the immediate FishBack metric and reduces future distributional distortion at the same immediate effect.

Neither route inherits a success claim from the other. They are not combined, named as a joint method, or used to reopen version-20 validation unless both pass their own prospective gates.

## Dependency graph

1. Complete the claim-relative nearest-work audit and constrain priority claims.
2. Freeze the mathematical estimand, data allocation, baselines, analysis, cost budget, and stop rules for each pilot.
3. Reuse existing Fisher, exact linear algebra, hashing, receipt, and clustered-interval primitives; add only missing robust-QP and autoregressive-Jacobian primitives.
4. Pass CPU algebra and fail-closed serialization tests.
5. Run one-state GPU smoke for each pilot.
6. Run independent exposed sentinel panels of 8-16 groups.
7. Open a powered pilot only for a route that passes every sentinel gate.
8. Promote only a passing pilot to a second architecture or broader behavior panel.
9. Re-run the evidence-strength audit. Paper sections remain blocked until at least one input-matched pilot is `fair-strong`.

## Faster experimental funnel

Every stage is cheaper than the next and has a prospective stop:

| Stage | Purpose | Maximum allocation | Promotion condition |
|---|---|---:|---|
| Algebra | Verify KKT, PSD nesting, exact solve, KL Taylor term, and invalid-input behavior | CPU only | All deterministic tests pass |
| Smoke | Verify model replay, hook/cache semantics, finite metrics, and atomic receipts | 1 group per pilot | Exact invariants and receipt checks pass |
| Sentinel | Estimate feasibility, reachability, runtime, memory, and effect direction without confirmatory language | 8-16 independent groups per pilot | Every premise method runs; deferred full-comparator methods remain blocked from development; no structural kill gate fails |
| Development | Select only declared layer, regularization, slack penalty, and operating point | 24 groups per dataset for A; 24 states for B | Frozen development gates pass |
| Pilot test | Estimate the predeclared primary effect and uncertainty on untouched groups | 48 groups per dataset for A; 48 states with 8 rollouts for B | Primary, retention, coverage, mechanism, and cost gates pass |
| Scaling | Test architecture and behavior breadth | Not authorized yet | A pilot has a complete fair-strong receipt |

Partial output never changes the next stage's method, thresholds, strata, or sample allocation. A group writes its result and receipt atomically; a comparator failure is serialized explicitly and does not abort unrelated groups. A required candidate failure fails that group and the corresponding coverage gate without selective reruns.

## GPU schedule

The current cluster check found seven of eight H100s on `avi-gn-fsk42` occupied by another user, while `avi-gn-fsk35`, `avi-gn-fsk40`, and `avi-gn-fsk41` each exposed eight empty H100 80GB cards and the project, GPT-2, Gemma-2-9B, and Python environment were present on every checked host. Capacity is rechecked immediately before each launch.

The intended schedule after both implementations pass locally is:

- four GPUs run independent InvariantBack group shards with one Gemma-2-9B replica per GPU;
- four GPUs run independent CacheBack state shards with one GPT-2 replica per GPU;
- remaining free hosts are not occupied before a sentinel demonstrates that additional parallelism reduces wall time rather than duplicating failed work;
- CPU preparation, receipt validation, and analysis run while GPU shards execute;
- the next immutable packet is prepared before the current packet finishes, but is not launched until the current promotion gate is evaluated.

This uses eight GPUs for independent scientific units instead of splitting one small solve across devices. A single-host fallback uses the same four-plus-four layout; a capacity-constrained fallback runs the two pilots sequentially without changing their estimands.

## Analysis depth

Every result packet reports more than a pooled mean:

- independent-unit and matched-pair definitions;
- attrition and missingness by method and stratum;
- cluster bootstrap intervals with frozen seeds and resample counts;
- dataset, target, encoding, layer, horizon, and reachability strata;
- target-effect, collateral-distribution, canonical-performance, and compute Pareto fronts;
- action norms, condition numbers, active constraints or slack, eigenvalues, angles, per-offset costs, wall time, peak memory, and exact derivative counts;
- negative controls, mechanism controls, and severe individual failures;
- gate-by-gate decisions with no substitution of a favorable secondary endpoint.

The analysis distinguishes a failed method from a failed implementation, and a local mechanism result from a behavioral or SOTA result.

## Promotion and stopping

InvariantBack is closed if hard constraints are systematically infeasible, if the predefined slack rule cannot restore coverage without missing the semantic gate, or if any gain is confined to construction encodings. CacheBack is closed or narrowed to a mechanism result if the future metric is numerically indistinguishable from `H=0`, the local quadratic prediction fails, the gain is single-offset, the immediate effect is not retained, or exact geometry is operationally impractical under the frozen budget.

If exactly one route passes, it becomes the central method and the other remains an adverse or mechanistic result. If both pass independently, a later document may propose a factorial joint test. No joint experiment is currently authorized.
