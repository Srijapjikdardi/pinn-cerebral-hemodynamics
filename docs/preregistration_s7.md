# Pre-registration: S7 on leave-one-vessel-out splits (written 2026-11-07, before any vessel_* run)

S7 (cycle-mean flow balance along each vessel, `reliability/goodhart.py`) was defined after seeing the projection results on the
`variant` and `complexity` splits. To test it on data it was not designed on, these runs and criteria are fixed in advance.

Runs: graph surrogate (width 128, 4 rounds, 5 members, 100 epochs, seed base 600), with and without `--project`, on splits
`vessel_ACoA`, `vessel_PCoA_L`, `vessel_A1_L`, `vessel_P1_L` (left side only; the network and priors are left-right symmetric).
Label: err_Q above the val 90th percentile. AUROC within held-out samples (test_ood).

Hypotheses (each judged per split; overall claim needs >= 3 of 4 splits):
- H7a: on the projected surrogate, S7 AUROC >= 0.70.
- H7b: on the projected surrogate, S7 AUROC exceeds S7 AUROC on the unprojected surrogate.
- H7c: on the projected surrogate, S7 median is >= 3x its unprojected median.
- H1-repl: on the projected surrogate, S1 AUROC is within 0.5 +- 0.1 after rounding S1 to 1e-7 (enforced check uninformative).
No other signal, label, threshold or split is used to judge these. If H7a/H7b fail on >= 2 splits, S7 is reported as split-specific.
