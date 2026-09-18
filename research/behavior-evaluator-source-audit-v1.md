# Behavior-evaluator source audit, version 1

Date: 2026-09-07. Decision: no replacement evaluator is admitted and no GPU route is opened.

## Sources and question

This audit asks whether public AxBench artifacts provide an external behavioral anchor that can replace the failed local Qwen C/I/F proxy without human or official GPT-4o-mini labels.

The inspected AxBench revision is `stanfordnlp/axbench@41c8332543e5a631f9a8c0a9df38799893ace758`, dated 2026-03-12. The pinned `rule_judge.py` SHA-256 is `46e1973f5561d3aa67b1e6dc70084a776707f8d119379fa8bef59934bbb61914`, `constants.py` is `39a1ef42bd1d7b4a59fd37a94ff36d90b3d844c281de619bce756362b1d2bb33`, and `human_eval.py` is `b047ce473c2b8a86769facc019654dcab0de75f92f6ce740aee6614c275c6134`.

The [AxBench paper](https://arxiv.org/abs/2501.17148) defines open-generation steering quality as the harmonic mean of LLM-rated Concept, Instruction, and Fluency scores. It does not report a human-agreement validation for that judge. The [official repository](https://github.com/stanfordnlp/axbench/tree/41c8332543e5a631f9a8c0a9df38799893ace758) includes a human-survey HTML generator that asks raters for the same three scores and downloads their answers for email submission, but recursive tree inspection found no checked-in human-response packet.

## Rule-evaluator contract audit

The official source maps 25 natural-language format instructions to deterministic or heuristic functions. It does not score general semantic concept incorporation. Several implemented contracts are materially weaker than the mapped instruction:

- `include mutliple email addresses` passes when one email is present.
- `include multiple telephone numbers` passes when one number is present.
- the numbered-list instruction passes on any single number followed by a period.
- the postscript instruction passes on a `P.S.` line anywhere, not necessarily at the end.
- the citation instruction passes on any URL rather than a citation and reference contract.
- JSON validity returns Boolean 0/1 while the surrounding evaluator uses a 0/2 scale.
- the two word-limit functions reference `nltk` without importing it.
- `use only capital letters` is mapped to a fraction of whitespace-delimited words whose `isupper()` is true; the separate exact all-caps function is not registered.
- paragraph-separation rules pass after observing one delimiter rather than verifying every paragraph boundary.

Language detection and passive/past-tense rules additionally depend on heuristic external analyzers. Even the source-faithful rules that execute correctly cover formatting or surface language, not Concept relevance, topical Instruction relevance, or Fluency.

## Decision

The public `RuleEvaluator` is not admitted as an external anchor for the failed C/I/F instrument. Selecting only convenient rules after this source inspection would narrow the target from open-ended semantic steering to format following and still would not validate Fluency or Instruction scoring. Repairing its contracts would create a project-specific evaluator rather than reproduce the public benchmark.

This is a source-level negative preflight, not an empirical claim that rule-grounded benchmarks are useless. A future narrow rule-following study may use independently specified executable tests, but it must state that different estimand prospectively. The current open-vocabulary route still requires either official GPT-4o-mini access with failure-aware receipts or an independently collected blinded human calibration packet before another method can use behavior scores for selection.
