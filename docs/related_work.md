# Related work: literature refresh (2026-11-07)

Searched 2026-11-07 (web search, abstracts and snippets; full texts not read unless noted). Earlier sweeps: 2026-11-05 and the
first session (see `docs/audit.md`, "Literature lock"). Check every claim against the PDF before citing. Do not write "first".

## 1. Physics residual as a trust signal / solver fallback (closest method overlap)

- **NeuroForge** (Jabbary et al., arXiv 2607.10333, v3 25 Aug 2026). Steady-RANS residual of the prediction as a trust signal;
  per-case rank correlation with field error positive across three backbones (Transolver, Geo-FNO, MeshGraphNet) and a second dataset.
  Residual fails as a correction objective (iterations lower error while raising residual). Uncertainty-gated classical fallback (stub
  in their code). https://arxiv.org/abs/2607.10333
  - Difference from us: CFD, not vascular networks; no topology-shift protocol; no hard-constrained surrogate; does not ask what happens
    to the signal when the model enforces it.
- **Hybrid Neural World Models** (arXiv 2605.28317). Error map from horizon self-consistency gates a solver fallback; cites
  concurrent residual- or RL-gated hybrid solvers (Roy et al. 2025, Srikishan et al. 2025). https://arxiv.org/abs/2605.28317
- **ANCHOR** (2512.19643), **XRePIT** (2510.21804): residual-triggered fallback in CFD (from earlier sweep).
- **Error-Conditioned Neural Solvers** (Jiang, Wang, Chen et al., arXiv 2606.27354). Feeds the residual back as an error signal;
  reports a residual-reconstruction gap where a test-time-optimised model's L2 error rises while its residual falls.
  https://arxiv.org/pdf/2606.27354

## 2. Low residual is not low error (supports our Goodhart framing)

- **Inverted model selection in PINNs** (Musah, arXiv 2608.21683, Aug 2026). The lower-residual model is the worse solution in most of
  64 matched pairs and across six architecture families; proposes an admissibility gate of independent structural checks.
  https://arxiv.org/pdf/2608.21683 . Closest conceptual neighbour to our finding; theirs is model selection on fixed problems, ours
  is per-case failure detection under topology shift with a hard constraint.
- **Silent failures in PINNs** (McShannon, Dietrich, arXiv 2606.25151). Poisoned physics parameters reach equal or lower residual loss
  with large errors; loss-based validation cannot detect them. https://arxiv.org/pdf/2606.25151
- Classical: residual needs stability weighting to bound error (Eiras 2024; Astral, arXiv 2406.02645).

## 3. Hard constraint enforcement by projection (what our junction layer is an instance of)

- Training-free projections onto governing-equation constraints (Rochman Sharabi, Louppe, arXiv 2511.17258); KKT-hPINN projection
  layers for hard linear equality constraints (arXiv 2402.07251); constraint-projected learning for conservation laws (arXiv 2511.03578).
  These report lower constraint violation and often lower error. None we found measures whether the enforced quantity can still be used
  to detect failures afterwards, or where the removed violation goes. That is our gap; state it as "we found no study", not "first".

## 4. Graph surrogates for vascular networks and unseen topologies

- Physics-informed GNNs for 1D blood flow (PubMed 39326359): graph from arterial topology, R^2 > 0.99 in silico.
- MeshGraphNet-based 1D reduced-order models trained on 3D data (Pegolotti et al., CBM 2024; arXiv 2303.07310); authors list
  generalisation to unseen geometries as future work.
- **GNN for cerebral blood flow on clinical datasets** (Kim et al., arXiv 2411.17971): predicts flow and pressure on cerebral vascular
  structures not seen in training (stenosis patients). Must be cited next to our graph surrogate; no reliability estimator or fallback.
- Microvascular physics-informed GNNs (Botta et al., arXiv 2512.10792).
- Zhao et al. (arXiv 2610.02238; found in the earlier sweep, not re-found today): reuse of a 55-segment model on pruned trees, no runtime
  error estimate. Re-verify the identifier before citing.

## 5. Circle of Willis modelling and ML surrogates

- Alastruey et al. 2007 (J Biomech 40:1794): geometry source and anatomical variants.
- Surrogate for UQ of CoW collateral flow (PLOS Comput Biol 2022, doi 10.1371/journal.pcbi.1009996): DNN surrogate of a 1D-0D model,
  no topology-shift reliability study.
- Sarabian et al. PINNs for cerebral hemodynamics (arXiv 2108.11498); Kaczmarek et al. neural operator transformers (CBM 2025).

## Position after this refresh

Unchanged and sharpened: residual-based trust and fallback exist (section 1), and "low residual is not low error" is established
(section 2). The defensible contribution is (i) a verified CoW benchmark with leave-topology-out splits, (ii) a systematic evaluation
of physics checks as failure detectors under topology shift for two surrogate families, (iii) the hard-constraint result: the enforced
check becomes uninformative and the removed violation reappears where the enforcement mechanism puts it (S7, pending pre-registered
replication on `vessel_*` splits), (iv) fallback cost accounting.
