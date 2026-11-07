# Pre-registration: second topology domain, pruned 37-artery network (written 2026-11-07, before any dataset or model run on it)

Purpose: test whether the CoW findings hold on a different vascular tree. Domain: 37 uniform segments from the Boileau 2015
benchmark-5 supplement (`scripts/make_network_boileau37.py`), seven side-branch roots removable (s3, s4, s9, s11, s16, s24, s31; removing
one prunes its subtree), 96 valid topologies, pure-resistor terminals, zeta = 9 friction, inflow from the supplement. Solver: dx 4 mm
(worst Q discretisation difference vs dx 1 mm: 1.3%, P 0.09%). Dataset: 1000 samples, seed 11, same perturbation spreads as the CoW,
terminal conductance scaling of inflow by retained outflow demand (1/R1), NX = 21, NT = 80.

Runs (same protocol as the CoW): splits `variant` and `complexity` (`data/splits.py`, unchanged definitions); graph surrogate (width
128, 4 rounds, 5 members, 100 epochs, seed base 600) without and with `--project`; MLP (4 x 256, 5 members, 100 epochs, seed base 100)
without `--project`. Label: err_Q above the validation 90th percentile. AUROC within held-out topologies (test_ood), topology-cluster
bootstrap 95% CI. Signals S1-S7 as defined in `reliability/`. Nothing else is tuned on this domain; model settings are the CoW settings.

Hypotheses (each judged per surrogate and split; a claim "holds" if it is met on both splits for the graph surrogate):
- H1 (check works): unconstrained S1 AUROC >= 0.70 and its cluster CI lower bound > 0.5.
- H2 (Goodhart): projected S1 AUROC within 0.5 +- 0.1 after rounding S1 to 1e-7.
- H3 (error migration): projected S7 median >= 3x unprojected S7 median.
- H4: projected S7 AUROC > unprojected S7 AUROC.
- H5 (detector bar, as in the CoW pre-registration): projected S7 AUROC >= 0.70 at the val-p90 label. Also reported: balanced label.
- H6 (fallback): graph, S1 gate, deferral of the top 30% of held-out cases by score removes more mean err_Q than deferring 30% at
  random (paired by split).
- H7 (architecture): graph OOD err_Q is lower than MLP OOD err_Q on both splits.
Failures are reported as failures. Out-of-scope for judging: any other signal, label, split or threshold.
If the 37-artery solver or data fail their own checks (non-convergence, balance > 1%, pressures outside 40-180 mmHg in more than 30% of
samples), the dataset is regenerated and the check reported; hypotheses are not changed.
