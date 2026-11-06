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

## Results (added after the runs; the text above is unchanged since commit 9e5058b)

Graph surrogate, held-out vessel absence. err_Q label val p90 as pre-registered. "bad%" = share of held-out cases above the threshold.

| split | OOD err_Q graph / projected | bad% (projected) | S7 AUROC unprojected -> projected | S7 median ratio | S1 projected, rounded 1e-7 |
| --- | --- | --- | --- | --- | --- |
| vessel_ACoA | 0.102 / 0.078 | 75% | 0.36 -> 0.92 | 11.9x | 0.50 |
| vessel_PCoA_L | 0.047 / 0.036 | 22% | 0.66 -> 0.82 | 4.8x | 0.50 |
| vessel_A1_L | 0.184 / 0.159 | 99% | 0.33 -> 0.66 | 15.2x | 0.50 |
| vessel_P1_L | 0.172 / 0.152 | 100% | undefined (no case below threshold) | 15.5x | undefined |

Verdict by the pre-registered rule (>= 3 of 4 splits):
- H7a (projected S7 >= 0.70): **not supported.** 2 pass (ACoA, PCoA_L), 1 fail (A1_L 0.66, AUROC there rests on ~8 "good" cases),
  1 undefined (P1_L).
- H7b (projected S7 > unprojected S7): **supported** on all 3 defined splits; P1_L undefined. Counted 3 of 4 only if an undefined
  split is not counted against it; we report it as "3 of 3 defined".
- H7c (S7 median >= 3x): **supported**, 4 of 4 (4.8-15.5x).
- H1-repl (projected S1 uninformative): **supported**, 3 of 3 defined.

Exploratory, not pre-registered (balanced label err > held-out median, defined on every split): projected S7 0.90 / 0.73 / 0.97 /
0.95 (ACoA / PCoA_L / A1_L / P1_L) vs unprojected S7 0.40 / 0.62 / 0.58 / 0.72; unprojected S1 0.91 / 0.82 / 0.97 / 0.96.

Reading: the mechanism (violation moves into the mean-flow balance, S7 rises 5-16x) replicates on every held-out vessel. The detector
claim at a fixed in-distribution threshold does not reach the pre-set bar, mainly because held-out A1 or P1 absence makes nearly every
case "bad" (error 4-6x the in-distribution level), which leaves AUROC undefined or resting on a handful of cases. Paper wording: S7 is
a mechanism result with strong ranking power, not a validated stand-alone gate.
