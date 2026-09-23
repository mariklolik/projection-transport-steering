# Confirmatory protocol v24: PTS in the setting of PCHI

Frozen and pushed before any rollout on the validation split exists.
Setting (after Li et al., 2026, arXiv:2606.09876): Qwen3-4B-Instruct-2507, the
JSON self-evaluation prompt of their Appendix A.1, greedy decoding (768 new
tokens), 10,000 unique OpenMathInstruct-2 problems (train_1M order), first 5,000
train and last 5,000 validation; responses whose JSON does not parse are
dropped. Groups WY/CY by the sign of the yes-no logit gap at the readout
position and exact-match correctness. Decision: calibrated diagonal-LDA probe on
the layer-18 readout state, fitted on train (70% fit, 30% calibration), fires
at p >= 0.5 (`results/pchi/probe.pt`). PCHI: 128 head coefficients at layers
19-22, trained on train with their stated hyper-parameters (200 steps, batch 8,
lr 0.04, L1 0.05, rho 0.7) and margin -1 (`results/pchi/heads.pt`).

Arms on validation, selected on train: PCHI; PTS = the decision gating a shift
of the readout state along the WY-to-WN mean difference at 8 times the mean
state norm (smallest dose of the train maximum); ungated shift at dose 2 (train
maximum); references: 100 random unit directions through the same decision at
dose 8, and 20 random permutations of the PCHI coefficients through the same
decision.

Hypotheses (paired bootstrap 10^4 over questions, Holm over 3):
- H1: Sel(PTS) > Sel(PCHI). H2: Sel(PTS) > Sel(ungated shift).
- H3: Sel(PCHI) > mean Sel of its permuted-coefficient nulls.
Reported: WY correction, CY damage, ECE and AUROC of the stated confidence for
every arm; the random-direction distribution for PTS.
