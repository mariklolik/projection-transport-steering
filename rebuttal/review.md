# Reviewer concern and manuscript response

## Concern

Validate the detection axis across more models and behavioral targets. Add one targeted confirmatory experiment with uncertainty estimates, a stratified error analysis, and limitations tied to each major claim. State the assumptions essential to the main claim and add a controlled stress test showing where the mechanism or bound stops holding.

## Manuscript changes

- The main text now reports the positive held-out OLMo-2/TruthfulQA confirmation.
- Table 4 reports adjusted paired BCa intervals against the norm-matched action and representative construction controls.
- Experimental Setup records the frozen allocation, model panels, resampling unit, multiplicity correction, and observer gates.
- Limitations and Assumptions separates operator optimality from behavioral sufficiency, states the Monge regularity requirement, distinguishes concavified ROC area from raw AUROC, and separates operator cost from the two-pass gate.
- The appendix records the complete stress-test boundary and label-stratified ETHICS result.

## Direct answer

The detector-axis action transfers beyond Gemma on OLMo-2/TruthfulQA: calibrated MC2 improves by +0.137 with adjusted 95% CI [+0.105, +0.168], and MC1 improves by +0.105 with adjusted 95% CI [+0.025, +0.177], relative to the norm-matched spherical action. All seven pre-frozen specificity and alignment controls pass. The paper does not claim universal transfer; the complete denominator and boundary cases remain in the appendix and released audits.
