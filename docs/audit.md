# Gate 0 audit

Date: 2026-11-05. Scope: solver, topology, shipped dataset, PINN/evaluation code (read, not re-run).

## Verdict

**Gate 0 solver verification: FAIL.** The bundled solver does not conserve mass, has no outlet
boundary conditions, and models the anterior communicating artery (ACoA) as a self-loop. The shipped
dataset (`data/synthetic_demo.h5`) and every checkpoint trained on it are not valid ground truth for
the reliability study. Keep them only as a pipeline smoke test.

Per the project plan's kill criterion ("solver fails verification and is not fixed: stop, fix solver,
push all dates"), the next work item is a verified solver, before any new data or training.

How this was checked: `pytest` (11 pass, 19 expected-failures that document the defects below),
`scripts/gate0/dataset_budget.py`, `scripts/gate0/solver_experiments.py`. Environment: 2-core CPU
container, Python 3.13, NumPy 2.5. PyTorch could not be installed (download.pytorch.org returned 403
through the proxy), so no model or checkpoint was run.

## Solver and data findings

Severity: C = invalidates ground truth, H = must fix before data generation, M = fix or document.

| ID | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| S1 | C | No outlet boundary conditions. Terminal ends (MCA distal, and any dead-end node) are never updated, so A and Q stay at initial values (A0, Q=0): a closed wall. | `generate_synthetic.py` lines 222-231 update only nodes with 2+ segments. MCA `Q[:, -1]` is exactly 0 in all 60 patients. |
| S2 | C | Volume is not conserved. Per patient, about 3.0 mL enters per cycle, 0 leaves, and only 0.5% (range 0.3-0.8%) is stored. | `dataset_budget.py` on `synthetic_demo.h5`. Patient 0, MCA_L: 1.014 mL in, 0 out, 0.0014 mL stored. |
| S3 | C | ACoA is a self-loop (`ACoA_mid` to `ACoA_mid`). A1_L and A1_R share that node, so left and right stay connected with ACoA absent. "Missing ACoA" is not an anatomical change. | `tests/test_topology.py` (two xfail tests). Removing ACoA still changes results (up to 21% in P, 73% in Q on A1) through the relaxation rule, not through a flow path. |
| S4 | C | Absent PCoA turns PCA_P1 into a closed tube fed by the basilar inflow. | PCoA_L absent: PCA_P1_L pressure differs by 99% from the complete network. |
| S5 | H | Wave-speed estimate is inconsistent with the tube law. Code gives sqrt(E h / (2 rho r0)); the tube law implies sqrt(2 E h / (3 rho r0)). Ratio 0.866 at every E and diameter. The time step is derived from the code value. | `test_wave_speed_estimate_matches_tube_law`. No instability seen in 60 patients; the lost CFL margin was not quantified. |
| S6 | H | Shipped data uses 6 points per segment. Peak pressure differs from a 41-point solve by up to 36.7% (one segment); at 21 points the worst difference is 0.4%. MCA and PCA waveform shapes do not converge (relative L2 vs 41 points at 21 points: MCA_L P 32%, PCA_P1_L Q 57%). | Mesh study below. Part of the waveform gap may come from S1/S4 dead-end reflections and the one-cycle window; unresolved. |
| S7 | H | Inlet calibration is undocumented. Total inflow is about 3.0 mL per 0.8 s cycle (about 225 mL/min), which looks low against typical total cerebral blood flow (roughly 700 mL/min; verify against Alastruey et al. 2007 before relying on it). | Dataset inflow volume. |
| S8 | H | Pressure reference is 0 at A0 (gauge). Peak pressure is 3.8-11.8 kPa (mean 6.1) while physiological arterial pressure is about 10-16 kPa. | Dataset `P`. |
| S9 | M | Demo configs run 1 cycle from rest (A=A0, Q=0), so the saved window is the start-up transient, not a periodic state. `default.yaml` runs 4 cycles. | `configs/demo.yaml`, solver loop. |
| S10 | M | Junction model is pressure averaging plus an equal split of the Kirchhoff residual. It is not characteristic-based and uses static, not total, pressure. At PCA_P1 junctions both vessels count as inflow, so sum Q = 0 forces Q_P1 = -Q_PCoA. | Lines 233-258; `network_topology.py`. |
| S11 | M | Junction Kirchhoff residual is about 1e-17 in the data because the solver enforces it by construction. It passes while S2 fails, so it cannot validate the solver. As a surrogate-failure signal it is still meaningful on predictions. | `test_junction_boundary_values_satisfy_kirchhoff`. |
| S12 | M | Only ACoA, PCoA_L, PCoA_R can be absent. No A1 or P1 variants, no A2/P2 distal vessels, no outlets. The study plan needs all of these. | `configs/*.yaml`, `network_topology.py`. |

What works: units are consistent SI; inlet waveform is applied as prescribed (test passes); output is
finite, areas positive, runs are deterministic.

## PINN and evaluation findings (from reading code and README; not re-run)

| ID | Sev | Finding |
| --- | --- | --- |
| P1 | H | Held-out patients are random draws, not held-out topologies. The only topology variation is 3 optional segments. |
| P2 | H | Reported metric is peak-pressure MAPE per segment. P is computed analytically from A in the model, so this is an area-error proxy. No waveform error for Q or A is reported. |
| P3 | H | README logs show about 20 resume and fine-tune rounds, keeping the lowest MAPE on the same held-out set. That is selection on the test set. |
| P4 | M | Best reported MAPE: 27.9% (20 patients) and 55.7% (60 patients) against the 5% target in `configs/default.yaml`. These were measured on the invalid ground truth above. |
| P5 | M | The model is already conditioned on a 23-value vector (2 x 11 + 1, checked in tests), so the plan's "Option A" (topology-conditioned surrogate) is partly in place. |

## Solver benchmark (Gate 0 part B)

Reference patient: baseline diameters, E = 8e5 Pa, demo settings (6 points per segment, 1 cycle).

| Setting | Wall time | Source |
| --- | --- | --- |
| Demo resolution, this container (2 cores) | 11.1 s mean, 0.2 s std, n=5 | `solver_experiments.py timing_demo 5` |
| Demo resolution, 60-patient dataset (author's machine) | 13.2 s mean (6.4-22.9) | `solver_time_s` attribute in the h5 |
| Default resolution (20 points, 4 cycles), this container | 142 s, n=1 (README claims about 450 s on the author's machine) | `solver_experiments.py timing_default` |
| 21 points, 1 cycle | 43.9 s | `solver_experiments.py mesh 21` |
| 41 points, 1 cycle | 87.4 s | `solver_experiments.py mesh 41` |

The plan's framing rule maps about 11 s to "seconds: reliability study, no speed claim". That rule
applies to the verified solver, not this one. Re-measure after the rewrite (a vectorised or compiled
solver may drop below 1 s, which would remove the efficiency argument for fallback).

Peak pressure, relative difference vs 41 points, worst segment: 6 points 36.7%, 11 points 7.7%,
21 points 0.4%.

## Literature lock (Gate 0 part C)

Done by a delegated search-and-read pass over 10 sources; the full-text claims are the agent's
extraction, so check each against the PDF before citing. Not readable: Kaczmarek 2025 beyond its
methods, Yuhn 2022 beyond its methods, NeuroForge section 7, one candidate (arXiv 2610.23774, rate-limited).

- No paper found combines a physics-residual check, solver fallback, and held-out CoW topologies for
  1D blood flow. This is a search result, not proof; do not write "first".
- Closest method overlap: NeuroForge (arXiv 2607.10333), ANCHOR (2512.19643), XRePIT (2510.21804).
  They establish residual-triggered selective prediction or fallback in CFD, so the contribution here
  is the domain and the topology shift, not the idea.
- Closest topology overlap: Zhao et al. (arXiv 2610.02238) reuse a 55-segment model on pruned
  47/43/35-segment trees without retraining (errors rise from about 1-2% to 3-9%). It has no runtime
  error estimator or fallback. Read it in full before the related-work section.
- Protocol changes to adopt: report a random-deferral baseline and an oracle; do not count deferred
  cases as zero error; per-variant AUROC with several seeds (AUROC can fall below 0.5 under shift);
  measure the residual floor on true solver output; include ensemble spread and a fused score;
  note conformal exchangeability fails under topology shift.
- Alastruey et al. 2007 (J Biomech 40:1794) Table 1 gives CoW lengths and radii (for example A1
  1.2 cm / 0.117 cm, P1 0.5 / 0.107, ACoA 0.3 / 0.074, PCoA 1.5 / 0.073, basilar 2.9 / 0.162) and
  covers absent or hypoplastic A1, P1, ACoA, PCoAs. Use it as the geometry source; "the standard"
  is unverified.

## Next: verified solver (proposed spec)

1. Conservative second-order scheme in (A, Q) form with the existing tube law and a wave speed
   derived from it.
2. Characteristic-based inlet (prescribed Q), junctions (mass plus total pressure, Newton solve),
   and RCR Windkessel outlets at every terminal (MCA, A2, P2).
3. Geometry from Alastruey 2007 Table 1, every number logged with its source in `configs/`.
   Variants as real graph changes (ACoA and PCoAs as two-node edges; A1 and P1 absent or hypoplastic).
4. Verification suite before any data: single-vessel wave speed and Poiseuille limit; bifurcation
   reflection coefficient vs analytic; global volume balance within 1e-3 per cycle; periodic state
   after N cycles; mesh convergence order; Boileau et al. 2015 cases once reference values are in hand.
5. When tests pass, remove the xfail markers (strict mode will force it) and re-measure timing.

Open decision: rewrite in-repo (Python, vectorised or numba; Colab-friendly, full control) versus
adopt an existing validated 1D solver. Default is the in-repo rewrite unless you prefer otherwise.
Effort estimate 2-4 weeks including verification; uncertain.

## Update 2026-11-05: verified solver timing

`physics/solver_1d.py` + `physics/cow_network.py` (commit 95f25ac); `scripts/gate0/time_cow.py`. Full 15-vessel CoW,
periodic state (5 cycles, tol 1e-5), this container (2 cores), numba, n=3 after JIT warm-up:

| dx | cells | wall |
| --- | --- | --- |
| 2 mm | 367 | 3.4 s |
| 1 mm | 722 | 5.2 s |

Max Courant 0.68. Not yet verified: mesh convergence on the full network, Boileau 2015 benchmark. Per the plan's
framing rule, 3-5 s is "seconds: reliability study, no solver-speed claim"; a PINN forward pass (ms) still gives a
speed gap, so the fallback-cost argument depends on surrogate latency, to be measured once torch is installed.

Mesh convergence, full CoW, periodic state (tol 1e-6, 7 cycles each), mid-vessel, worst vessel, vs dx = 0.5 mm
(`scripts/gate0/cow_mesh_study.py`; Q error denominator floored at 1% of the largest vessel because symmetric ACoA Q is ~0):

| dx | Pmax diff | P rel L2 | Q rel L2 (worst: PCoA_L) | wall |
| --- | --- | --- | --- | --- |
| 4 mm | 0.005% | 0.027% | 0.15% | 3.5 s |
| 2 mm | 0.001% | 0.019% | 0.29% | 4.6 s |
| 1 mm | 0.001% | 0.006% | 0.14% | 6.7 s |
| 0.5 mm (ref) | - | - | - | 22.8 s |

Reference is the finest grid, not an exact solution, so this shows grid-independence at 4 mm, not accuracy. The small
Q errors in PCoA_L do not fall monotonically (small-flow vessel, few cells). Use dx = 2 mm for dataset generation
(about 4.6 s); revisit for hypoplastic variants.

## Update 2026-11-05: dataset v1 and data-only baseline

`data/cow_v1.h5` (seed 1, gitignored, 455 MB): 1500 samples, all 96 valid topologies (7-25 samples each), all converged,
max Courant 0.73, median 6 cycles to periodic state. 1031 of 1500 (69%) pass the `physiological` flag
(systolic <= 180 and diastolic >= 40 mmHg); systolic p90 218, p99 270 mmHg. Inflow scaling, pulsatility 0.5 and
spreads are assumptions (see `data/generate_cow.py`, `configs/geometry_alastruey2007.yaml`).

Baseline: data-only MLP (4 x 256, topology mask + physiology -> full A, Q fields), no physics loss
(`training/train_surrogate.py`, 150 epochs, 2 seeds, model selection on val only). Mean relative L2, Q / A, per sample:

| split | val | test_id | test_ood |
| --- | --- | --- | --- |
| random | 6.3% / 7.2% | 6.3% / 6.9% | - |
| variant (25% of topologies held out) | 7.0% / 8.0% | 7.2% / 8.6% | 10.0% / 10.6% |
| complexity (train <=2 absent, test >=4) | 11.3% / 14.0% | 11.4% / 13.9% | 22.4% / 21.2% |

Reading: held-out topologies cost about 3 points (7.2 -> 10.0%); extrapolating to more absent vessels costs
about 11 points (11.4 -> 22.4%). The surrogate degrades under topology shift but not catastrophically, so the reliability signal has
to separate a modest error increase, not a collapse. Two seeds only; no error bars yet. Not yet evaluated: any
reliability signal (S1-S6), fallback, vessel-held-out splits.

## Update 2026-11-05: first reliability-signal results (kill check passed)

5-member data-only MLP ensemble per split (100 epochs, seeds 100-104), ensemble-mean prediction. Label "bad" = err_Q above
the 90th percentile of validation (in-distribution) err_Q. AUROC within the held-out-topology subset (`test_ood`);
code: `reliability/signals.py`, `training/reliability_eval.py`, `reliability/metrics.py`; tables in `outputs/rel_*.npz`.

| signal | variant test_ood | complexity test_ood |
| --- | --- | --- |
| S1 junction mass | 0.73 | 0.90 |
| S2 junction total-pressure mismatch | 0.59 | 0.63 |
| S3 PDE residual (finite differences) | 0.62 | 0.72 |
| S4 inlet-flow mismatch | 0.70 | 0.91 |
| S5 ensemble spread | 0.48 | 0.74 |
| S6 input kNN distance | 0.61 | 0.64 |
| fused (logistic, fit on val only) | 0.70 | 0.86 |

AURC (mean err_Q of retained cases, coverages 10-100%), variant test_ood: random 0.093, oracle 0.065, best signal
(fused) 0.078, so about half the random-to-oracle gap. Complexity test_ood: random 0.241, oracle 0.178, S4 0.185, S1 0.193.

Reading:
- Kill criterion (best AUROC near 0.5 under topology shift) is not triggered: physics signals carry real information.
- Ensemble spread is uninformative for held-out topologies in `variant` (0.48): members share data and inductive bias and fail together. It helps only in `complexity` (0.74), where the extrapolation is large. S6 (input distance) is weak in both.
- S1 and S4 dominate. Caution: S4 compares the predicted inlet flow with the prescribed inflow, so it is close to a direct
  error measurement on one boundary of the same quantity (Q) the label is built from. It is legitimate at deployment
  (inflow is a known input) but should not be sold as an independent physics check. S1 is likewise Q-based. The label uses
  Q only; repeat with an A- or P-based label before claiming generality.
- Residual floor on exact solver output: S1 1.6e-4, S2 7.5e-4, S4 3.0e-4, S3 0.245 (S3 is high even on truth because grid-point
  finite differences on a 21-point grid are coarse; it separates predictions, whose median is about 1.0, but its absolute
  value is not interpretable). Predicted medians are 100-300x the floor for S1/S2/S4.
- Limits: one dataset, one surrogate family, one seed set; thresholds from a single quantile; no confidence intervals;
  pooled AUROC mixes distribution shift with error ranking, so the within-OOD numbers are the honest ones;
  `physiological`-flagged cases not separated; no calibration or fallback yet.

## Update 2026-11-05: Boileau 2015, partial (case 1 only)

Source read: Boileau et al., Int J Numer Meth Biomed Eng 31(10), e02732 (Swansea open-access PDF). Only the text up to Table V
was readable, so Tables VI-XIII (bifurcation, 37-artery, ADAN56 geometry) were not available, and the paper's
supplementary reference data (stated "publicly available online") was not located; no URL appears in the extracted text.
Cases 2-4 (carotid, thoracic aorta, aortic bifurcation) compare against 3-D simulations that are not in hand. Case 5
(37 silicone arteries) needs Table IX plus the in vitro data of Matthys et al. Not verified: any of these.

Verified: case 1, single pulse in a 10 m reflection-free tube, inviscid (`tests/test_boileau_single_pulse.py`).
Paper parameters used as given: A0 = pi cm^2, h = 1.5 mm, E = 400 kPa, rho = 1050, Qin = 1e-6 exp(-1e4 (t-0.05)^2).
Our tube law constants coincide with the paper's (beta = (4/3) sqrt(pi) E h at sigma = 0.5), giving c0 = 6.17 m/s as reported.
Result vs the linear solution Q(x,t) = Qin(t - x/c0), relative L2 of Q at 5 m / 9.7 m: dx = 2 mm 0.6% / 1.3%; peak
ratio 1.000; pressure peak within 3% of Z0 Qmax. At dx = 4 mm 7% / 12% (diffusion, peak 0.97).
At dx = 1 mm the error rises again (1.9% / 4.4%): the pulse arrives about 0.5 ms early, converging to that value under
refinement. This is the nonlinear advection speed u + c(A) (u/c0 about 5e-4 at the peak), which the linear reference omits,
not a solver error; so the linear reference sets a floor of about that size and this test cannot resolve finer.
The viscous variant is not tested: our friction closure (K_R = 8 pi mu / rho, Smith 2001) differs from the paper's
profile-dependent term (zeta = 9), so a pass or fail there would test the closure choice, not the solver.
Status: Gate 0 solver verification is analytic limits plus one published benchmark case, not the full benchmark.

## Update 2026-11-06: robustness of the reliability results

`reliability/robustness.py`; two independent 5-member ensembles per split (seed bases 100, 200; same data and split).
AUROC within held-out topologies, mean over the two ensembles (spread between ensembles <= 0.03 for every entry).
Bootstrap 95% CIs (first ensemble, resampling test samples; optimistic because samples within a topology are correlated).

| signal | variant, Q label | variant, A label | complexity, Q label | complexity, A label |
| --- | --- | --- | --- | --- |
| S1 junction mass | 0.74 | 0.75 | 0.90 | 0.66 |
| S2 junction pressure | 0.60 | 0.56 | 0.65 | 0.53 |
| S3 PDE residual | 0.64 | 0.53 | 0.69 | 0.63 |
| S4 inlet mismatch | 0.73 | 0.71 | 0.92 | 0.61 |
| S5 ensemble spread | 0.49 | 0.46 | 0.74 | 0.65 |
| S6 input distance | 0.63 | 0.58 | 0.64 | 0.61 |

"A label" = area error above the validation 90th percentile; pressure is a monotone function of area in this model, so
this also stands for pressure. Findings:
- Results are stable across ensembles; the earlier numbers were not seed luck (at n = 2 ensembles).
- On `variant`, S1 and S4 hold up under the A label (0.75, 0.71). On `complexity` they fall (0.66, 0.61): their large
  advantage there is specific to Q error. Under the A label on `complexity` no signal exceeds 0.66.
- S2 (junction total pressure) is weak under every label (0.53-0.65), including the A label, which is the quantity it is built from.
- S3 on `variant` with the A label is at chance (0.53, CI includes 0.5).
- Ensemble spread is at or below chance on `variant` (0.46-0.49): not a usable signal for held-out topologies of this size.
- Claim supported so far: physics-based junction-mass and inlet-flow signals predict flow-error under topology shift
  (AUROC 0.73-0.92); evidence for pressure/area error is weaker (0.53-0.75). Do not generalise beyond one surrogate family.

## Update 2026-11-06: solver-fallback policy (first cost-accuracy results)

`reliability/fallback.py`. Policy: surrogate always runs; a case is sent to the verified solver when its score exceeds a
threshold calibrated on validation (in-distribution) only. Deferred cases get error 0 by construction (dataset solver
output is the ground truth), so this evaluates the policy, not the solver. Costs: solver 2.49 s/case (dataset `wall_s`,
2 parallel workers; 4.6 s single-process at 2 mm), 5-member surrogate 9-12 ms, signals 4 ms: speed ratio 150-190x.

Variant split, held-out topologies (n = 386), mean err_Q: surrogate only 0.093; deferring the top 30% by score gives 0.058 (S1),
0.058 (S4), 0.059 (fused) vs random 0.065 and oracle 0.049; at 50%: 0.038 / 0.041 / 0.041 vs random 0.046, oracle 0.031.
Val-calibrated threshold (q = 0.80): deferral rises from 0.20 on test_id to 0.44 on held-out topologies and the error
reduction is 54% (S1) against 44% for random deferral at the same rate.
Complexity split (n = 599): a val-calibrated threshold defers 82-98% of held-out cases (vs 3-27% in distribution) and cuts
error 88-99%, but random deferral at the same rate cuts 82-98%, so almost all of that reduction is volume.

Reading:
- The checker's clearest practical value is automatic shift detection: deferral climbs by 2-30x from in-distribution to
  held-out topologies without ever seeing them.
- The gain over random deferral is real but modest: roughly 5-10 points of error reduction on `variant`, 3-6 points on
  `complexity` at val-calibrated operating points. Oracle headroom is about twice that.
- With a solver at 2.5 s, deferring most cases costs about as much as always running it (2.4 s/case at 97% deferral). The
  efficiency claim (surrogate saves compute) holds only for in-distribution traffic. Under strong shift the result is a
  safety mechanism, not a speed-up. Frame the paper around reliability, not speed.
- Deferred error is 0 only because the dataset solver defines truth; real fallback inherits solver error (0.6-1.3% in
  the Boileau single-pulse test at 2 mm).
- Only err_Q; fused score fit on val with a Q label; no conformal guarantee (exchangeability fails under topology shift).

## Update 2026-11-06: Boileau 2015 benchmark 5 (37-artery network) verified

Data: the paper's supplementary zip for benchmark 5 (supplied by you): `mesh.txt` (geometry, PWV, terminal R), `inflow.txt`,
`ExpData` (in vitro pressure and flow at 8 sites, Matthys et al. 2007), `NumData` (six codes: DCG, LCG, FEM, FVM, McC, STM).
Third-party data is gitignored (`data/boileau37/`). Code: `scripts/gate0/boileau37_run.py`, `boileau37_compare.py`.
Setup from the supplement and Matthys 2007: rho 1050, mu 2.5 mPa s, E from supplied PWV via our tube law (gives 1.2 MPa,
Matthys' measured value), terminals single resistors R1 (mesh col 10 x 1e9; parallel resistance 2.248e8 Pa s/m3 = Matthys'
total peripheral resistance), p_out = 3.2 mmHg (Matthys), inflow from `inflow.txt` (mean 52.0 ml/s). Tapered segments are
chains of uniform sub-vessels (<= 1 cm). Run to periodic state (about 17 cycles), last cycle compared at the paper's time windows.
Site-to-segment mapping is not in the supplement; inferred by matching to the six-scheme mean (not to in vitro data;
best-match margins 0.07-0.21 vs next-best 0.11-0.49): AorticArchII 10, ThoracicAortaII 17, LSubclavianI 11, RIliacFemoralII 29,
LUlnar 14, RAnteriorTibial 34, RUlnar 7, Splenic 20, evaluated at segment midpoints. Not independently confirmed.

Findings (relative RMS over the cycle, normalised by RMS of the reference):
- Flow split is right from the start: mean flow in segment 10 is 37.2 ml/s vs 36.95 (FVM).
- Harness bug found and fixed during this work: terminal-capacitor state initialised at 12 kPa instead of p_out doubled mean pressure.
- With K_R = 8 pi mu / rho (parabolic profile) the pressure error vs in vitro is 2.0-6.5% vs 1.7-3.7% for the six codes;
  distal flow error up to 49% vs 41%. Mesh refinement (dx 2 -> 1 mm, sub-vessel 1 -> 0.5 cm) changes nothing (<=3 points), so
  the gap is model, not discretisation.
- Setting friction to the paper's velocity-profile parameter zeta = 9 (2 pi mu zeta^2/(zeta-1) = 2.53 x 8 pi mu, chosen from the
  paper's stated zeta for the aorta cases before running, not fitted) closes the gap:

| site | P error vs in vitro (ours / six-code mean) | Q error vs in vitro (ours / six-code mean) | ours vs six-code mean, P / Q |
| --- | --- | --- | --- |
| AorticArchII | 1.6% / 1.7% | 28.3% / 28.2% | 0.4% / 1.1% |
| ThoracicAortaII | 1.9% / 2.1% | 55.3% / 57.6% | 0.4% / 2.6% |
| LSubclavianI | 3.4% / 3.2% | 29.8% / 30.2% | 0.4% / 1.6% |
| RIliacFemoralII | 4.1% / 3.7% | 40.7% / 41.2% | 0.4% / 2.3% |
| LUlnar | 2.7% / 2.6% | 15.1% / 15.3% | 0.3% / 2.0% |
| RAnteriorTibial | 3.3% / 3.4% | 15.3% / 14.5% | 0.6% / 1.9% |
| RUlnar | 2.7% / 2.6% | 14.1% / 14.2% | 0.3% / 0.9% |
| Splenic | 2.5% / 2.3% | 13.2% / 12.8% | 0.4% / 1.3% |

Our solution sits within 0.3-0.6% (P) and 0.9-2.6% (Q) of the six-code mean; the six codes agree with each other to
0.1-0.4% (P) and 0.9-2.0% (Q). Large flow errors against the in vitro data (up to 55%) are shared by all codes, so they reflect
the experiment (flow-probe measurement and model idealisation), not this solver.

Consequences:
- Gate 0 solver verification now includes a published multi-vessel benchmark with in vitro data, at near-parity with the six
  reference codes. State it as: "reproduces benchmark 5 of Boileau et al. 2015 to within about 2x the spread among the six
  reference codes". Cases 2-4 (3-D reference) remain unchecked.
- IMPORTANT: the CoW dataset uses K_R = 8 pi mu / rho (parabolic) with mu = 4.5 mPa s following Alastruey 2007's Smith 2001
  closure; the benchmark suggests zeta = 9 is the common closure for 1D codes. This choice changes pressure by a few percent
  and flow by more. Decide deliberately; if the paper claims a physiologically faithful solver, document the closure and test sensitivity.

## Update 2026-11-06: Boileau 2015 benchmark 1 (single pulse) against the six codes

Data: supplement `Benchmark1_SinglePulseModel` (supplied by you; pressure vs distance at t = 0.1-1.5 s for six codes).
`scripts/gate0/boileau1_run.py`; relative L2 of pressure profiles against the six-code mean:

| case | six-code spread about their mean | ours vs six-code mean |
| --- | --- | --- |
| inviscid | 1.61% | 1.88% |
| viscous, mu = 4 mPa s, friction 8 pi mu (parabolic) | 2.01% | 22.36% |
| viscous, friction scaled for zeta = 9 (x2.53) | 2.01% | 2.90% |

Peaks at t = 0.1 / 0.7 / 1.5 s (kPa), viscous: reference 0.0201 / 0.0157 / 0.0112; zeta = 9: 0.0202 / 0.0161 / 0.0118;
parabolic 0.0204 / 0.0187 / 0.0165. The inviscid case matches the six codes within their own spread. The viscous case
matches only with the zeta = 9 closure; the parabolic closure used for the CoW dataset decays too slowly (peak at 1.5 s
0.0165 vs 0.0112 kPa) over a 10 m tube. CoW vessels are at most 12 cm, so the effect there is much smaller, but it is not
zero: this strengthens the open decision on the friction closure noted above (test sensitivity or regenerate with zeta = 9).
Cases 2, 4, 6 (carotid, aortic bifurcation, ADAN56; data on hand) not yet compared.

## Update 2026-11-07: friction-closure sensitivity on the CoW dataset

`scripts/gate0/friction_sensitivity.py`: 150 dataset samples (seed 1, same parameters) re-solved with friction_scale 1 (parabolic,
dataset) and 2.53 (zeta = 9). All converged. Relative difference over the periodic cycle (median / p90 / max, %):

| field | median | p90 | max |
| --- | --- | --- | --- |
| A | 0.30 | 0.37 | 0.52 |
| Q | 1.74 | 3.01 | 7.07 |
| P (gauge) | 6.00 | 7.46 | 9.26 |
| P max | 11.5 | 15.8 | 21.9 |
| P min | 3.0 | 6.0 | 10.8 |

Reading: flow and area barely change (Q 1.7% median, below the surrogate error of 7-10%), but gauge pressure moves 6% in L2 and
11% at the systolic peak, because P is a steep function of small area changes. Any pressure-based claim made on the parabolic-closure
dataset would carry that bias; flow-based reliability results are less affected. Given two benchmarks favour zeta = 9, regenerate the
dataset with it before final results. (Not fitted to the CoW; chosen from the benchmark paper.)

## Update 2026-11-07: Boileau 2015 benchmark 2 (common carotid artery) and a correction on the friction closure

Data: supplement `Benchmark2_CommonCarotidArtery` (inflow.txt; NumData: 3-D reference and six 1D codes, each 1D file holds the whole run,
so the last cycle is the window t = 8.8-9.9 s from the README). `scripts/gate0/boileau2_run.py`. Setup from the paper (Tables I, III):
L 126 mm, rd 3 mm (our A0 := Ad, pressure offset Pd = 10.933 kPa), h 0.3 mm, E 700 kPa, rho 1060, mu 4 mPa s, Windkessel R1 2.4875e8,
C 1.7529e-10, R2 1.8697e9, Pout 0, **zeta = 2** (the paper's value for this case = our default friction 8 pi mu / rho).
The 3-D file is shifted by 0.055 s relative to the 1D time axis (inflow.txt: "add 0.055 to one period to have the minimum flow at t=0");
I applied exactly that shift; fitting the shift against the six-code mean (not our solution) gives 57 ms, so 2 ms residual.
Relative RMS over the cycle at the midpoint (normalised by RMS of the reference):

| quantity | ours vs 3-D | ours vs six-code mean | six-code spread about their mean |
| --- | --- | --- | --- |
| pressure | 0.31% | 0.02% | 0.02% |
| flow | 0.67% | 0.08% | 0.04% |
| radius change | 1.84% | 0.09% | 0.10% |

Ours agrees with the six 1D codes to within about their own spread, and with the 3-D reference to under 1% for P and Q (the paper
reports under 1% for all three; our radius-change figure is 1.8% in this normalisation, which I did not try to reconcile).
With zeta = 9 on this case the agreement worsens (P 0.50%, dr 2.03% vs the six codes): the paper uses zeta = 2 for the carotid.

CORRECTION to the earlier friction note: the paper sets zeta per test case (9 for the aorta cases and the single pulse, 2 for the
carotid). I wrote that "the benchmark suggests zeta = 9 is the common closure"; that over-generalised. Cerebral arteries are small
vessels like the carotid, so the parabolic closure used for the CoW dataset (zeta = 2) is the better-supported default and matches the
carotid benchmark. The CoW dataset therefore stays as it is; no regeneration is needed for this reason. The sensitivity run above
(zeta 2 -> 9: Q 1.7%, P 6% median) stays as a bound on the closure uncertainty and should be quoted as such.

## Update 2026-11-07: topology-cluster bootstrap

`reliability/cluster_bootstrap.py`: resample held-out topologies (24 in `variant`, 38 in `complexity`), then samples within them;
mean AUROC over the two ensembles. The earlier sample-level CIs were too narrow, as expected.

| signal | variant Q-label | variant A-label | complexity Q-label | complexity A-label |
| --- | --- | --- | --- | --- |
| S1 junction mass | 0.74 [0.59, 0.85] | 0.75 [0.66, 0.83] | 0.90 [0.84, 0.94] | 0.66 [0.56, 0.74] |
| S2 junction pressure | 0.60 [0.48, 0.72] | 0.56 [0.47, 0.68] | 0.65 [0.56, 0.73] | 0.53 [0.45, 0.62] |
| S3 PDE residual | 0.64 [0.54, 0.73] | 0.53 [0.42, 0.64] | 0.69 [0.58, 0.79] | 0.63 [0.55, 0.70] |
| S4 inlet mismatch | 0.73 [0.64, 0.83] | 0.71 [0.59, 0.81] | 0.92 [0.87, 0.96] | 0.61 [0.49, 0.72] |
| S5 ensemble spread | 0.49 [0.39, 0.61] | 0.46 [0.33, 0.61] | 0.74 [0.64, 0.83] | 0.65 [0.57, 0.73] |
| S6 input distance | 0.63 [0.56, 0.70] | 0.58 [0.51, 0.66] | 0.64 [0.56, 0.73] | 0.61 [0.54, 0.68] |

Still above chance for S1/S4 on every split and label except S4 with the A label on `complexity` (CI includes 0.5). S2, S3 (A label)
and S5 (`variant`) include 0.5 in their intervals. Only S1 and S4 can be called informative.

## Update 2026-11-07: physics-regularised surrogate (partial answer to the "do signals survive PINN training" threat)

`training/physics_loss.py` adds continuity (A_t + Q_x by finite differences) and junction-mass penalties to the data loss; same
architecture and protocol, 5-member ensembles (`outputs/relphys_*.npz`, seed base 400). Weight chosen from two trial values on a
1-member 50-epoch `variant` run (val relative L2: w = 0.03 -> 0.118, 0.3 -> 0.170, 3 -> 0.278; data-only about 0.08 at 100 epochs), so
0.03 was picked for accuracy, not tuned for the reliability result. Within held-out topologies, AUROC (S1/S4 shown; err_Q label / err_A label):

| surrogate | variant S1 | variant S4 | complexity S1 | complexity S4 | OOD err_Q (variant / complexity) |
| --- | --- | --- | --- | --- | --- |
| data-only (2 ensembles) | 0.73-0.75 / 0.75 | 0.70-0.76 / 0.70-0.72 | 0.89-0.90 / 0.65-0.66 | 0.91-0.93 / 0.59-0.63 | 0.093-0.096 / 0.241-0.246 |
| physics-regularised w=0.03 | 0.73 / 0.62 | 0.73 / 0.72 | 0.83 / 0.65 | 0.91 / 0.61 | 0.098 / 0.258 |

Findings, with limits:
- The weak regulariser did not do what a Goodhart test needs: the median junction-mass signal on held-out topologies is unchanged
  (variant 0.0245 vs 0.023-0.024; complexity 0.0725 vs 0.072-0.074), i.e. the surrogate still violates junction conservation about
  100x more than the exact solver (1.6e-4). Accuracy is slightly worse (OOD err_Q 9.8% vs 9.3-9.6%).
- Because the residual was not driven down, this run cannot show whether signals survive a surrogate that satisfies the physics.
  It does show that adding these penalties at this weight leaves the signals informative (S1 Q-label 0.73 / 0.83, S4 0.73 / 0.91).
  Under the area label, S1 on `variant` drops from 0.75 to 0.62 while S2 rises from 0.55 to 0.63 (single ensemble; not interpreted).
- A decisive test needs a surrogate that actually satisfies the constraints: a stronger or better-scheduled penalty (the MLP trades
  accuracy for it, w = 0.3 cost 5 points), a hard-constrained output (junction-conserving parametrisation), or a different
  architecture. Not done. Until then the paper cannot claim robustness to physics-informed training.
