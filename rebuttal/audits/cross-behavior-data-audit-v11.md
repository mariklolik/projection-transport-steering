# Cross-behavior data audit, method version 11

Audit date: 2026-09-05. No model output was observed during this audit.

## Candidate-panel decision

The frozen Anthropic sycophancy source was not selected as a primary panel. The NLP-survey and PhilPapers files at revision `d533f626cc321c92175a58ee570aa3cdb87238d1` are byte-identical, with SHA-256 `582860b42e2beec806a7d361a08bcce7fdb264e2e697d55104074540352fb308`. The nominal 9,984 NLP rows contain only 32 distinct claim-and-choice clusters of 312 persona variants each. The 10,200 political rows contain 15 distinct choice-pair clusters, 13 of size 600 and two of size 1,200. Treating rows as independent would understate uncertainty; using 47 source clusters would be too weak for the primary breadth claim.

BBQ and ETHICS were selected because they supply thousands of independent labeled units, natural response alternatives, exact open revisions, and established construct definitions. Neither uses new human or model-generated labels in this project.

## BBQ integrity

The exact 11 source JSONL files contain 58,492 rows. The official pinned target-location table omits 16 rows; applying the official merge rule retains 58,476 rows forming 14,619 complete four-row templates. Every retained template has one ambiguous and one disambiguated context under both negative and nonnegative question polarity.

The frozen allocation retains exactly 20/10/30 templates per category for representation fit, observer fit, and confirmatory evaluation. This yields 220/110/330 independent templates and 880/440/1,320 question rows. Confirmatory category counts are exactly 120 rows and 30 clusters for each of 11 categories. Answer-position counts are not forced: the confirmatory labels select alternatives 0/1/2 on 424/458/438 rows.

The BBQ split manifest SHA-256 is `dcf2e27a4f9bb635d939540782bc902943d9a3721c134345fac51f4d59b8c16c`.

## ETHICS integrity

The pinned Hugging Face loader maps `train.csv` to train, `test.csv` to validation, and `test_hard.csv` to test. Only source rows with `is_short=True` are eligible, preventing silent truncation of long scenarios at the fixed sequence budget. The eligible pools contain 6,661 train, 2,109 validation, and 1,704 hard-test rows; their longest inputs contain 49, 33, and 46 whitespace-delimited words. The frozen train allocation contains 600 representation-fit rows and 300 disjoint observer rows. Confirmatory evaluation contains 500 stable-hash rows from validation and 500 from hard test.

Desired labels are 352/248 for representation fit, 171/129 for observer fit, 286/214 for validation confirmatory, and 264/236 for hard-test confirmatory. Both polarity strata are therefore represented in every stage without outcome-dependent resampling.

The ETHICS split manifest SHA-256 is `18f52cc2a83adb53488a5937878f8609d220a5d82d96c91ca4a5696775098167`.

Pinned-tokenizer preflight over all allocated prefixes and answer alternatives found maximum complete sequence lengths of 198/177/214 tokens for BBQ and 89/79/95 for ETHICS on Mistral/OLMo-2/Granite. All are below the frozen 512-token boundary, so no selected row requires truncation or exclusion.

## Decision

The dataset and allocation gate passes. BBQ inference must resample four-row templates, not rows. ETHICS inference uses rows and reports validation, hard-test, and label-polarity strata. This receipt authorizes observer output under benchmark version 18 but does not authorize confirmatory output for a cell that fails its observer screen.
