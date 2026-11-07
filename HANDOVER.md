# Handover: CoW surrogate reliability project (state at 2026-11-07)

For the next AI assistant or collaborator. Read this first, then `docs/audit.md` (every number, chronological) and
`docs/pipeline_review.md` (threats and their status). Do not trust any number in this file over `docs/audit.md`.

## 1. Goal and constraints set by the project owner

- Owner: Mannat Mangla (4th-year Biomedical Engineering, TIET Patiala). Wants (a) a publishable research paper, (b) code good enough to
  show to foreign universities and companies for research positions in biomedical AI.
- Working rules the owner asked for: make decisions that serve a strong paper, explain the reasons, then **verify every decision**
  (build, test, check, fix, repeat). Discuss real trade-offs with the owner. Be honest: report negative results, never claim "first".
- Owner's style preference in chat: very terse replies ("caveman" mode), but documents in plain, simple English. No em dashes.
- GitHub repo: `https://github.com/Srijapjikdardi/pinn-cerebral-hemodynamics`, working branch **`gate0-audit`** (main = legacy
  prototype). The assistant could not push (GitHub not linked); the owner pushes from the zip: `git push origin gate0-audit`.
- Supervisors: Dr. Vishal Srivastava, Dr. Vishal Gupta (owner decides who reviews; venue/author order not yet decided).
  Earlier plan notes mention a "Dr. Malhotra" for venue sanity checks; confirm with the owner.

## 2. Research question and framing (agreed)

"Can cheap physics-consistency checks computed from a neural surrogate's own prediction flag its large errors on Circle of Willis (CoW)
topologies not seen in training, and what happens when the surrogate is built to satisfy those checks?"

- Method novelty is low by itself: residual-as-trust and residual-triggered solver fallback exist (NeuroForge arXiv 2607.10333, ANCHOR
  2512.19643, XRePIT 2510.21804, Hybrid Neural World Models 2605.28317). "Low residual is not low error" is also known (Musah 2608.21683).
- Defensible contribution: (i) verified 1D CoW benchmark with leave-topology-out splits; (ii) systematic evaluation of 7 checks, two
  surrogate families, cluster-bootstrap CIs; (iii) **Goodhart result**: a hard junction-conservation layer makes its own check (S1)
  uninformative while errors remain, and the removed violation reappears in the vessel interior where check S7 detects it (mechanism
  replicated in a pre-registered test; S7 as a fixed-threshold gate did not meet the pre-set bar); (iv) honest fallback cost accounting.
- Paper type: methods paper on verified-solver synthetic data. No clinical claims. Wording: "we found no study that ...".
- Target venues (realistic): Computers in Biology and Medicine; Int J Numer Methods Biomed Eng; Biomech Model Mechanobiol; Ann Biomed
  Eng; plus arXiv preprint. Workshop (ML4H / AI4Science) as fallback. Top ML venues and JCP judged unlikely.

## 3. What is done (key numbers; full detail in docs/audit.md)

| Item | Result | Where |
| --- | --- | --- |
| Legacy solver audit | failed: no outlet BCs, no mass conservation, ACoA self-loop; legacy data/checkpoints invalid | audit "Gate 0" |
| New solver `physics/solver_1d.py` (numba FV, Lax-Wendroff, characteristic inlet/junction, RCR outlets) | volume balance to round-off; mesh-converged at 2 mm | tests, audit |
| Boileau 2015 case 1 (single pulse) | inviscid within six-code spread | `scripts/gate0/boileau1_run.py` |
| case 2 (carotid) | vs six codes 0.02-0.09% (their spread 0.02-0.10%); vs 3-D 0.3% P, 0.7% Q | `boileau2_run.py` |
| case 5 (37-artery, in vitro) | within about 2x the six codes' spread (zeta = 9 for that case) | `boileau37_run.py`, `boileau37_compare.py` |
| case 6 (ADAN56) | P median 0.19% vs spread 0.35%; Q 1.28% vs 2.17%; DCG flow file is corrupt (dropped by a solution-independent rule) | `boileau6_run.py` |
| cases 3, 4 | not done. Case 3 (upper thoracic aorta): owner has the supplement zip, not yet sent/run. Case 4 (aortic bifurcation): zip sent earlier but needs paper Table VII (geometry), not readable here | |
| Friction closure | zeta = 2 for cerebral vessels (matches carotid case); zeta 2 -> 9 changes Q 1.7%, P 6% (median) | audit |
| Dataset `data/cow_v1.h5` | 1500 samples, 96 topologies (7 variable vessels), randomised physiology; 69% physiological pressures; regenerate, not in git | `data/generate_cow.py` |
| Splits | `random`, `variant` (25% topologies held out), `complexity` (train <= 2 absent, test >= 4), `vessel_<V>` | `data/splits.py` |
| Surrogates | MLP (4x256); vessel-graph message passing (width 128, 4 rounds, global readout); optional soft physics loss; optional hard junction projection | `training/train_surrogate.py`, `models/graph_surrogate.py`, `training/physics_loss.py` |
| Accuracy, OOD flow rel. L2 | MLP 9.3% variant / 24% complexity; graph 4.1% / 21%; graph worse than MLP at 6 absent vessels (0.55 vs 0.37 median) | audit |
| Signals | S1 junction mass, S2 junction total pressure, S3 PDE residual, S4 inlet mismatch, S5 ensemble spread, S6 input kNN, S7 cycle-mean flow balance, S8 projection correction size | `reliability/signals.py`, `reliability/goodhart.py` |
| Detection (AUROC within held-out topologies, flow label, cluster CI) | MLP: S1 0.74 [0.59, 0.85], S4 0.73 variant; graph: S1 0.95 [0.91, 0.98], S5 0.89 | `reliability/cluster_bootstrap.py` |
| Fallback | graph S1 gate: 74% error reduction vs 52% random at same deferral (variant); risk-coverage near oracle; solver 2.3-2.5 s/case vs ~14 ms surrogate+checks | `reliability/fallback.py` |
| Goodhart | hard projection: S1 -> round-off (1e-8 < solver floor 1.6e-4), AUROC -> 0.5 on both architectures and all splits; error falls 5-24%; S7 rises 5-18x and reaches AUROC 0.74-0.94 | audit, fig 3 |
| Pre-registered S7 test (`docs/preregistration_s7.md`, committed before runs) | mechanism H7b/H7c supported; H7a (S7 >= 0.70 at val-p90 label) not met (2 pass, 1 fail, 1 undefined) | fig 5 |
| Sensitivity | physiological subset (graph unchanged, MLP weaker); 4 label rules; balanced label shows pressure ranking 0.81-0.86 on complexity | `reliability/sensitivity.py` |
| Figures 1-5 | PNG+PDF in `figures/`, validated palette | `scripts/make_figures.py` |
| Literature refresh | `docs/related_work.md` (search snippets only; verify against PDFs before citing) | |

Not done / open: second topology domain; Boileau 3, 4; right-side `vessel_*` splits (symmetry assumed); soft-physics-loss Goodhart test
inconclusive (penalty too weak); S7 needs an independent validation before being claimed as a stand-alone gate; paper draft.

## 4. Environment notes (learned the hard way)

- 2-core CPU, ~8 GB RAM container. Torch from PyPI only; `download.pytorch.org` is blocked by the proxy. Working install: `pip download`
  the torch wheel and its CUDA deps in parallel into a folder, then `pip install --no-index --find-links <folder> torch` (~3.3 GB).
- Other deps: `pip install --break-system-packages numba h5py pyyaml tqdm scikit-learn pytest scipy matplotlib`.
- Regenerate data: `python data/generate_cow.py --n 1500 --out data/cow_v1.h5 --seed 1 --workers 2` (~30 min). Bit-identical to the
  original (checked against truth signals in committed npz files).
- Run at most **two** training jobs at once (each peaks ~3-4 GB); three caused OOM kills. Use `OMP_NUM_THREADS=1 PYTHONPATH=.`.
- In the chat harness, background jobs die a few minutes after a reply ends: keep long runs inside one working turn, launch via a
  script file (`nohup script.sh &`), and poll.
- `outputs/*_pred.npz` (float16 predictions, ~25-35 MB each) are gitignored; `goodhart.py` and figures need them or the small
  `*_decomp.npz` caches (committed). Note: S1 read from float16 preds is quantisation-limited (~2e-3); use the main npz for S1.
- Third-party benchmark data lives in gitignored `data/boileau37/`, `data/adan56/` (owner has the supplement zips for all six cases; 3 was never sent).
- `for_me_only/` is the owner's private explainer folder, gitignored. Do not publish it.
- Tests: `pytest -q` -> 36 pass, 19 strict xfails (they document legacy-solver defects; keep them).
- One-command rerun of the study: `bash scripts/reproduce.sh` (several CPU hours).

## 5. Plan for what comes next, and how

### Step A. Second topology-shift domain: pruned 37-artery network (recommended before writing; ~1 working day)

Why: reviewers will ask whether results hold beyond one 16-vessel ring. Generality is the main gap between workshop and journal tier.

How:
1. **Generalise the network description.** Today `data/cow_dataset.py`, `reliability/signals.py`, `training/physics_loss.py`,
   `models/graph_surrogate.py` read the CoW layout from `configs/geometry_alastruey2007.yaml` (`GEO`, `VESSELS`, `NODES`). Add a second
   config `configs/network_boileau37.yaml` with the same schema (vessels with length/radius/h/E, nodes as lists of (vessel, end),
   inlets, outlets with R/C, variable vessels) built from `data/boileau37/mesh.txt` (see `scripts/gate0/boileau37_run.py::build` for the
   parsing, taper chains and units). Make the four modules take the config path (env var or argument) instead of a module-level global.
   Keep CoW the default so every existing result reproduces; add a test that CoW outputs are unchanged.
2. **Define topology variants.** Choose ~6-7 side branches whose removal keeps the tree connected (for example arm branches, renal,
   one iliac side branch, splenic/hepatic). Removing a terminal branch removes its outflow; to keep physiology plausible, either keep
   total peripheral resistance fixed by rescaling the remaining terminals, or accept the redistribution and document it. Decide and
   record in the config before generating data. Verify the complete network still reproduces benchmark 5 after the refactor.
3. **Dataset.** New generator mode (`data/generate_cow.py --network boileau37`), ~1000-1500 samples, randomised r/E/inflow/R scales as
   for CoW, fixed grid per vessel (NX = 21, NT = 80). Time one solve first (37 vessels; boileau37 took ~17 cycles); budget accordingly.
4. **Experiments (same protocol, same seeds as CoW):** MLP and graph surrogates, 5 members, `variant` and `complexity` splits, with and
   without `--project`; signals S1-S7; cluster bootstrap; fallback; Goodhart decomposition. **Pre-register** the hypotheses for this
   domain first (copy the format of `docs/preregistration_s7.md`): S1 informative unconstrained; S1 -> chance with projection; S7 rises
   with projection; S7 >= 0.70 (balanced and val-p90 labels both reported). Commit before running.
5. Add a figure panel per domain and a table row per domain; update README, audit, pipeline_review.

### Step B. Paper draft (start methods/dataset now; results after Step A)

Suggested outline:
1. Introduction: fast vascular surrogates, silent failure under anatomical variation, need for runtime trust; contributions (4 bullets
   as in section 2 above).
2. Related work: residual trust and fallback; low-residual-not-low-error; hard constraint projection; vascular GNN surrogates
   (`docs/related_work.md`).
3. Methods: solver and verification (Boileau 1, 2, 5, 6 table); CoW geometry and variants (Alastruey 2007); dataset and physiology
   assumptions; splits; surrogates; projection layer; signals S1-S7 definitions and floors; metrics (AUROC within held-out set,
   topology-cluster bootstrap, risk-coverage, val-calibrated deferral); pre-registration.
4. Results: accuracy under shift (fig 4); detection (fig 1); fallback (fig 2); Goodhart and error migration (fig 3); pre-registered
   vessel-holdout test (fig 5); second domain; sensitivity (physiological subset, labels).
5. Discussion: when to trust which check; recommendation "check the quantity the enforcement mechanism perturbs"; limits (synthetic
   truth, assumed physiology, flow vs pressure, S7 post hoc, extreme-topology failure of graph model).
6. Reproducibility statement (code, seeds, `scripts/reproduce.sh`).
Make the draft as an editable document for the owner; keep every number traceable to `docs/audit.md`.

### Step C. Before submission

- Read the full PDFs of every cited paper (current notes are from abstracts/snippets). Re-verify arXiv IDs (Zhao 2610.02238 was not
  re-found in the last search). Check Boileau Table XI values used for ADAN56.
- Optional strengtheners if time: Boileau case 3 (ask the owner for the zip) and case 4 (needs Table VII); a third seed for the graph ensembles; right-side vessel splits;
  stronger soft-physics-loss schedule to complete that Goodhart arm.
- Owner shows draft to supervisors; decide venue and authors; arXiv + journal submission.

## 6. Decision log (why things are the way they are)

- Rewrote the solver instead of training on legacy data: legacy data was physically invalid (audit Gate 0).
- zeta = 2 friction for CoW (cerebral-size vessels match the carotid benchmark); zeta = 9 only for the aorta cases as in the paper.
- Labels are relative to each surrogate's own in-distribution error scale (val p90); balanced OOD-median label reported as exploratory.
- AUROC computed within held-out topologies (pooled AUROC would mix shift detection with error ranking).
- Graph architecture choices (width 128, global readout) picked on validation only; test sets never used for tuning.
- S7 was defined after seeing projection results; therefore a pre-registered replication was run and its partial failure reported.
- Speed claims dropped for shifted traffic (solver is only ~2.5 s); fallback framed as safety.
