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

## Results (added after the runs; the text above is unchanged since commit 7c2daba)

Data checks first (as pre-registered): 1000 samples, all converged, max Courant 0.68, 96 topologies (4-19 samples each), flow balance
(inlet vs summed terminal cycle-mean flow) median 6e-5, max 1.6e-4, 997 of 1000 samples inside 40-180 mmHg. Solver grid: dx 4 mm.
Two bugs were found and fixed before any experiment: terminal capacitors of pure-resistor outlets started at 12 kPa (mean pressure
doubled; fixed with `pc_init`), and `reliability/signals.py` assumed the CoW's dx, viscosity and friction (made config-driven).

Runs: graph (width 128, 4 rounds), projected graph, MLP; 5 members, 100 epochs, same protocol as the CoW; seeds 600 / 600 / 100.
Source: `reliability/prereg_b37.py`, `outputs/b37_prereg_results.json`, `outputs/b37_prereg.log`.

| | variant | complexity |
| --- | --- | --- |
| OOD err_Q graph / projected / MLP | 0.121 / 0.125 / 0.125 | 0.326 / 0.370 / 0.470 |
| ID err_Q graph / MLP | 0.120 / 0.119 | 0.157 / 0.149 |
| share of held-out cases above val-p90 (graph) | 8% | 94% |
| H1 unconstrained graph S1 AUROC >= 0.70, CI lower > 0.5 | 0.78 [0.58, 0.93] **pass** | 0.38 [0.18, 0.59] **fail** |
| H2 projected S1 (rounded 1e-7) within 0.5 +- 0.1 | 0.50 **pass** | 0.50 **pass** |
| H3 projected S7 median >= 3x unprojected | 6.2x **pass** | 12.0x **pass** |
| H4 projected S7 AUROC > unprojected | 0.71 vs 0.73 **fail** | 0.87 vs 0.31 **pass** |
| H5 projected S7 AUROC >= 0.70 (val-p90 label) | 0.71 **pass** | 0.87 **pass** |
| H6 S1 gate at 30% deferral removes more error than random | 0.114 vs 0.121 **pass** | 0.316 vs 0.326 **pass** |
| H7 graph OOD err_Q < MLP | 0.121 vs 0.125 **pass** | 0.326 vs 0.470 **pass** |

By the pre-registered rule (a claim holds if met on both splits for the graph surrogate): **H2, H3, H5, H6, H7 hold; H1 and H4 do not.**

What this does and does not support:
- The Goodhart mechanism replicates on a different vascular tree: the enforced check carries no information (H2) and the removed violation
  reappears in the mean-flow balance, 6-12x above the unprojected level (H3). This is the strongest cross-domain result.
- The detector claim does not transfer cleanly. H1 fails on `complexity`: the unconstrained graph's S1 is anti-correlated with error
  (0.38), while the MLP's S1 reaches 0.90 and S4 0.85-0.97 there. AUROC below 0.5 under extrapolation is a known failure of residual checks.
  Which check works depends on the surrogate and the domain.
- H5 passes at the val-p90 label but is fragile: with the balanced label (error above the held-out median) projected S7 is 0.63 / 0.59
  and the unprojected S7 is 0.64 / 0.47. On `complexity` 94% of held-out cases are "bad" at val-p90, so that AUROC rests on about 6% of cases.
- H6 passes by a small margin (6% and 3% lower kept-case error than random at 30% deferral); not a practical gain in this domain.
- The topology shift is mild in `variant`: held-out error equals in-distribution error (0.121 vs 0.120), so there is little to detect.
  Only 8% of held-out cases exceed the in-distribution p90.
- Accuracy is poor in absolute terms here: in-distribution flow error is 12-16% (area error 2%), against 2-5% for the graph surrogate on the
  CoW. CoW hyperparameters were reused, as pre-registered, and nothing was tuned; a longer-trained or wider surrogate may change the picture.
  This limits what the detector results can say about "good" surrogates.
