# Pipeline review against the publication goal (2026-11-07)

Goal: a paper with a defensible novelty claim that reviewers would accept. This review lists what is established, what threatens
the claim, and what we do about each. Evidence is in `docs/audit.md`.

## Claim we can currently support

"On 1D Circle of Willis blood-flow surrogates, physics-based junction-mass and inlet-flow checks computed at inference predict
flow-prediction error on topologies not seen in training (AUROC 0.73-0.92 within held-out topologies; 0.53-0.75 for area/pressure
error), and a deferral policy built on them detects the shift automatically and reduces error vs random deferral, at the cost of
extra solver calls." Verified solver (analytic limits, Boileau cases 1, 2, 5; case 4 and 6 pending).

Literature position (searched 2026-11-05 and 2026-11-07): no paper found that combines runtime physics-consistency scoring, solver
fallback, and held-out vascular topologies in 1D blood flow. Nearest: NeuroForge, ANCHOR, XRePIT (residual-triggered fallback in
CFD, other domains); Zhao et al. arXiv 2610.02238 (topology generalisation by pruning, no reliability estimator); arXiv 2610.27130
(3-D aorta surrogates; PDE residual as soft constraint and "a posteriori error estimator" suggested, no failure-detection evaluation,
no fallback); PUNCH arXiv 2601.17192 (residual only as a training loss, no shift test). This is a search result, not proof.
Novelty is domain plus topology shift plus a systematic evaluation, not the idea of residual-based gating.

## Threats, ranked

| # | Threat | Severity | Status / fix |
| --- | --- | --- | --- |
| 1 | Only one surrogate family (data-only MLP). Reviewers will ask whether signals survive a physics-informed surrogate (trained to satisfy the very residuals we use to detect failure; cf. Eiras 2024, residual != error). Project title says PINN. | High | Add a physics-regularised surrogate (continuity + junction loss) and a second architecture; report whether S1/S3 keep power. Result is publishable either way. |
| 2 | Signal strength is Q-specific. S1/S4 AUROC fall to 0.61-0.66 under area/pressure labels on `complexity`; S2 (junction pressure) is weak everywhere; S4 is close to a direct inlet-error measurement. | High | Report both labels everywhere; present S4 as a boundary-consistency check, not independent physics; do not claim generality for pressure. Improve pressure signals or say they fail. |
| 3 | Modest practical gain. Over random deferral: 5-10 points on `variant`, 3-6 on `complexity`; oracle headroom about twice that. | Medium | Report honestly. Frame value as shift detection (deferral rises 2-30x on held-out topologies) and safety, not speed. |
| 4 | Efficiency argument is weak: solver 2.5 s/case (4.6 s single process), so heavy deferral costs as much as always solving. | Medium | Drop the speed claim for shifted traffic; keep it for in-distribution (150-190x). |
| 5 | Statistical rigour: CIs resample samples, not topologies (optimistic); 2 ensembles; `vessel_*` splits unused; `complexity` test is 86% "bad" by the ID threshold. | Medium | Add topology-cluster bootstrap, 3+ ensembles, leave-one-vessel-out splits, a threshold defined relative to the surrogate's own error scale. |
| 6 | Ground truth is synthetic with assumed physiology (inflow waveform and magnitude, pulsatility 0.5, demand-based inflow scaling, perturbation spreads). 31% of samples fall outside 40-180 mmHg. | Medium | Verified solver reduces solver risk, but the inputs are assumptions. State as such; report results on the `physiological` subset; add sensitivity to key assumptions. |
| 7 | Single system. A reviewer may ask if results transfer beyond one 15-vessel network. | Medium | Optional extension: the Boileau 37-artery network with pruned branches as a second topology-shift domain (solver already reproduces it). |
| 8 | Solver verification incomplete: Boileau 3 (aorta, 3-D ref) not supplied; 4 (bifurcation) needs Table VII geometry; 6 (ADAN56) not yet run. | Low-Med | Need the paper PDF for Table VII (fetch blocked here). ADAN56 can be run from the supplement. |
| 9 | Repo hygiene: legacy README (about 900 lines of old logs), legacy checkpoints and `generate_synthetic.py` dataset still present and misleading. | Low | Replace README with the verified pipeline; mark legacy as deprecated; add `requirements` for numba/torch/sklearn; one-command reproduction. |
| 10 | Friction closure: zeta = 2 matches the carotid benchmark (cerebral-like vessels); zeta = 9 is the paper's choice for aortic cases. Sensitivity bound: Q 1.7%, P 6% (median). | Low | Keep zeta = 2; quote the bound. |

## What would stop publication

- If a physics-regularised surrogate makes S1/S3 useless (Goodhart), the paper becomes "residual gating fails for PINN-trained
  surrogates; here is when it works" — still publishable, but the claim changes. We must run this before writing.
- If no signal beats random deferral under area/pressure labels on a second surrogate, the claim must be limited to flow.
- If a reviewer demands clinical or in vivo validation, we cannot meet it with this data; the paper must be scoped as a methods
  study on verified-solver synthetic data.

## Decisions needed from you

1. Venue level: methods workshop / mid-tier journal (feasible with the plan above) versus a top biomedical-AI venue (would need
   in vivo or independent-solver validation we do not have).
2. Provide the Boileau paper PDF (or Table VII) if benchmark 4 is wanted.
3. Approve the physics-regularised surrogate as the next experiment (about 1.5 h compute).

## Status update 2026-11-07 (session 2)

- Threat 1 (one surrogate family; PINN-style surrogates): addressed. Second architecture (vessel-graph surrogate, 2x lower held-out error)
  and a hard junction-projection variant of both. Result: the enforced check (S1) becomes uninformative, the others survive, and the
  displaced inconsistency is detectable (S7). Soft physics loss (w = 0.03) remains inconclusive as before.
- Threat 2 (Q-specific signals): partly addressed. On the graph surrogate S1/S2/S5 reach 0.76-0.82 under the area label on `variant`;
  on `complexity` the apparent weakness is mostly label imbalance (balanced ranking label 0.81-0.86).
- Threat 3 (modest gain): addressed for the graph surrogate (S1 gate 74% error reduction vs 52% random at the same rate on `variant`;
  risk-coverage on the oracle curve).
- Threat 5 (statistics): topology-cluster CIs, two ensembles per model, four label definitions. Still open: `vessel_*` splits.
- Threat 6 (synthetic physiology): physiological-subset results reported; graph unchanged, MLP weaker. Still open: sensitivity to inflow
  and pulsatility assumptions.
- New caveat: S7 was defined after seeing the projection results; validate before claiming it as a general detector.
- Still open: threats 7 (second domain), 8 (Boileau 4, ADAN56), `vessel_*` splits, figures, literature refresh.
