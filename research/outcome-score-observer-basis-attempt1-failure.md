# Outcome-score observer basis attempt 1 failure

Status: failed before scientific artifact persistence.

The frozen `basis_completion` launch assigned four basis questions and rollout indices 2 and 3 to each of eight H100 workers. All workers reached the post-generation replay lookup and raised `StopIteration`. The runner selected the replay reference with a hard-coded `generation_id` suffix `:0`, but this stage contains only suffixes `:2` and `:3`.

The eight `failure.json` files are identical with SHA-256 `8e56f65990b80a56cec1949244ba2ddc964edd463555589c3e65a015da1abab3`. Their exception message is empty because `str(StopIteration())` is empty; the worker logs preserve the exception type and source line. No `rows.json`, `traces.pt`, or `receipt.json` exists in attempt 1, so it provides no admissible model result and cannot be analyzed or combined with the accepted v5 class-mix source.

The minimal correction derives the replay index from the first frozen `rollout_indices` entry. The regression test requires index 2 for `basis_completion` and index 0 for `fit`. It failed before the correction with a missing helper, then the scoped suite passed 7 tests, the full suite passed 257 tests with one unrelated warning, and critical Ruff checks passed.

Attempt 1 remains immutable at `/home/mekashirskiy/projection-transport-steering/artifacts/development/outcome_score_observer_v1_basis_completion_attempt1`. A corrected launch must use `outcome_score_observer_v1_basis_completion_attempt2`; calibration, validation, development, pilot, and confirmation remain closed.
