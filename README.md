> New collaborator or AI assistant: start with [`HANDOVER.md`](HANDOVER.md).

# When do physics checks catch neural-surrogate failures? Circle of Willis hemodynamics under anatomical variation

Neural surrogates for 1D blood flow are fast, but nothing tells you when they are wrong. This repository asks a narrow,
testable question: **can cheap physics-consistency checks, computed from a surrogate's own prediction, flag its large errors on
Circle of Willis (CoW) topologies it never saw in training, and what happens when the surrogate is built to satisfy those checks?**

Everything here is reproducible from code: a verified 1D solver, a 96-topology dataset generator, leave-topology-out splits,
two surrogate families, six reliability signals, a solver-fallback policy, and the statistics.

## Main findings so far

Evidence and every number below: [`docs/audit.md`](docs/audit.md). Threats and their status: [`docs/pipeline_review.md`](docs/pipeline_review.md).

1. **Topology shift hurts, but does not collapse, the surrogate.** Data-only MLP, relative L2 of flow: 7.2% on unseen physiology
   of seen topologies, 10.0% on held-out topologies (`variant`), 22.4% when extrapolating to more absent vessels (`complexity`).
2. **Two physics checks carry real signal under shift.** Within held-out topologies, junction mass conservation (S1) and inlet-flow
   consistency (S4) reach AUROC 0.73-0.92 for flow error (topology-cluster bootstrap CIs exclude 0.5). Pressure/area error is
   harder (0.53-0.75). Ensemble spread is at chance on `variant` (0.46-0.49).
3. **Goodhart effect, and where the error goes.** Add a hard junction-conservation layer to either surrogate: S1 falls to
   round-off (1e-8, far below the exact solver's 1.6e-4 floor) and stops detecting errors, while accuracy improves slightly
   (OOD flow error -5 to -15%). The inconsistency is not removed but moved into the vessel interior: a cycle-mean flow-balance
   check (S7) rises 5-18x and reaches AUROC 0.74-0.94 on all four model/split pairs. Checks the model does not enforce keep
   their power. A pre-registered replication on leave-one-vessel-out splits confirmed the mechanism (S7 rises 5-16x on all four held-out
   vessels) but not S7 as a fixed-threshold gate (2 of 4 splits reach AUROC 0.70; see `docs/preregistration_s7.md`).
4. **Better surrogate, stronger checks.** A vessel-graph surrogate (`models/graph_surrogate.py`) halves held-out-topology
   flow error (4.1% vs 9.3%). On it, S1 reaches AUROC 0.95 [0.90, 0.98], ensemble spread becomes informative (0.90), and an
   S1-gated fallback tracks the oracle. Results hold on a second ensemble and on the
   physiological-pressure subset. Extrapolation to >= 4 absent vessels stays hard for both families (18-24% OOD flow error).
5. **Fallback is a safety mechanism, not a speed-up, under shift.** The solver takes 2.5 s per case; a val-calibrated gate defers
   2-30x more often on held-out topologies without ever seeing them, and beats random deferral by 5-10 points of error reduction.

Not claimed: clinical validity (ground truth is a verified solver with assumed physiology), novelty of residual-based gating
itself (see related work in `docs/audit.md`), or results for pressure as strong as for flow.

## Repository layout

| Path | What it is |
| --- | --- |
| `physics/solver_1d.py` | 1D finite-volume solver (numba): Lax-Wendroff interior, characteristic inlet/junction/RCR outlet, round-off volume balance |
| `physics/cow_network.py`, `configs/geometry_alastruey2007.yaml` | CoW network builder; geometry from Alastruey et al. 2007, every number sourced |
| `data/generate_cow.py`, `data/splits.py` | 96 topology variants x randomised physiology; `random`, `variant`, `complexity`, `vessel_<V>` splits |
| `training/train_surrogate.py` | MLP and graph surrogates, optional physics loss or hard junction projection |
| `models/graph_surrogate.py` | Vessel-graph message-passing surrogate (second architecture) |
| `reliability/` | Signals S1-S6, AUROC / risk-coverage, cluster bootstrap, fallback policy, Goodhart decomposition |
| `scripts/gate0/` | Solver verification: mesh study, timing, Boileau et al. 2015 benchmarks 1, 2, 5 |
| `tests/` | 36 tests plus 19 strict xfails that document defects of the legacy solver kept for the audit trail |
| `docs/legacy_README.md`, `models/gpinn.py`, `data/generate_synthetic.py` | Legacy G-PINN prototype. Its solver failed verification (see audit); kept for history only |

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                                    # solver verification + unit tests (~1 min)
python data/generate_cow.py --n 1500 --out data/cow_v1.h5 --seed 1 --workers 2   # ~30 min on 2 cores, 455 MB
bash scripts/reproduce.sh                                    # surrogates, signals, tables (several CPU hours)
```

Boileau et al. 2015 benchmark data are third-party (paper supplement) and not redistributed; place them under
`data/boileau37/` etc. to rerun `scripts/gate0/boileau*_run.py`.

## Status

Work in progress toward a methods paper. Solver verification: analytic limits, mesh convergence, Boileau 2015 cases 1, 2, 5, 6
(cases 2 and 6 within the six reference codes' own spread). Pending: Boileau cases 3 and 4, a second topology-shift domain.
Figures: `python scripts/make_figures.py` -> `figures/`. Related work: `docs/related_work.md`.
