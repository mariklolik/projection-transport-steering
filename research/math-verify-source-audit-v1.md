# Math-Verify source and conformance audit, version 1

Audit date: 2026-09-07. Status: admissible only under the pinned strict configuration below.

## Source identity

- Package: `math-verify==0.9.0` from PyPI, released 2026-01-10.
- Project: `huggingface/Math-Verify`.
- Python requirement: 3.10 or newer.
- License: Apache-2.0; installed license SHA-256 `7625cd605224aba59452e3b2915e0afe8597c5f0bdf3cb670c9ac1852789f00b`.
- PyPI wheel SHA-256: `3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194`.
- Installed metadata SHA-256: `b26446cb88004aed0140885b0de51d811ffb155b735b300ead0e40461c5b2c6b`.
- Installed parser source SHA-256: `ea527a5a2403d97109a616bc92b7cc153320d6cecf4f823c070c727f5b77b5d0`.
- Installed grader source SHA-256: `8a789b7a1706333039cb9e3154c651d1dab4fc5d62c16aae0792f546a42a888a`.

The closest LRS release imports `parse` and `verify` from this package and evaluates the full model output. The package documentation describes mathematical-expression extraction and symbolic verification; it is not an LRS-specific evaluator.

## Default-config risk

The version-0.9.0 default combines LaTeX and plain-expression extraction, permits extraction without an explicit answer anchor, and keeps a first string fallback. In a targeted adversarial check, the unfinished text `unfinished reasoning says 504 but no final answer` parses as 504 and verifies against gold 504. That default is not admissible for terminal outcome reward because it can credit an intermediate or merely mentioned number.

## Frozen strict configuration

Prediction parsing uses:

- `LatexExtractionConfig(try_extract_without_anchor=False)` as the only extraction target;
- `fallback_mode="no_fallback"`;
- `extraction_mode="first_match"`;
- `parsing_timeout=5`;
- `raise_on_error=True`.

Gold answers use the package's default parse on the canonical dataset integer. Verification uses `strict=True`, `timeout_seconds=5`, and `raise_on_error=True`. Any exception, empty prediction parse, or nonfinite runtime fails the packet rather than becoming an incorrect label.

This configuration parses an anchored or boxed LaTeX answer but rejects bare numbers and unfinished prose. It returns one final candidate on all inspected version-4 rows.

## Conformance evidence

On the eight immutable version-4 Qwen2.5 outputs, the strict configuration parses 8/8 rows and verifies exactly 3 correct and 5 incorrect. The extracted candidates are `307, 1504, 1, 799, 15, 50, 44, 88`, matching the visible terminal boxes. This is post-result evaluator conformance only; it does not relabel or reopen version 4.

Adversarial checks show:

- a prose mention of 504 followed by final boxed 1504 verifies false against 504;
- an intermediate boxed 504 followed by corrected boxed 1504 verifies false;
- a final boxed 504 followed by trailing prose verifies true;
- unfinished prose containing 504 with no boxed answer yields an empty prediction parse.

## Operational boundary

Parsing and verification run serially on CPU after generation. The package uses signal-based timeouts and is not invoked from threads. Linux and macOS are in scope; the reported Windows multiprocessing limitation is outside this execution environment.

The evaluator is admissible for a new prospective gate only after the exact dependency lock, configuration, wrapper tests, and receipt fields are frozen. No existing output becomes confirmatory evidence through this audit.
