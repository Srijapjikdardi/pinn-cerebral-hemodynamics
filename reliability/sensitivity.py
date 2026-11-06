"""Sensitivity of the AUROC results to (a) the sample subset and (b) the definition of a "bad" prediction.

Subsets of held-out-topology samples: all; `physiological` only (systolic <= 180 and diastolic >= 40 mmHg in the exact solution).
Labels: err > val quantile q for q in {0.75, 0.90, 0.95} (relative to each surrogate's own in-distribution error scale), and
err > 2 x val median, and err > median err of the subset itself (pure ranking test, for splits where almost every
held-out case exceeds the in-distribution thresholds). Topology-cluster bootstrap CI as in cluster_bootstrap.py. Several files = independent ensembles, averaged.
Usage: PYTHONPATH=. python reliability/sensitivity.py outputs/relgraph_variant.npz [outputs/relgraph_variant_s700.npz]
"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from data.splits import load_topologies

SIGS = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch", "S5_ensemble_spread", "S6_input_ood"]
SHORT = ["S1", "S2", "S3", "S4", "S5", "S6"]
RULES = [("val p75", lambda v: np.percentile(v, 75)), ("val p90", lambda v: np.percentile(v, 90)),
         ("val p95", lambda v: np.percentile(v, 95)), ("2x val med", lambda v: 2 * np.median(v)),
         ("OOD median", None)]  # evaluation-only ranking label (half bad by construction); not usable at deployment


def cluster_ci(ds, taus, label, s, groups, keys, rng, nboot=500):
    bs = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([rng.choice(groups[keys[k]], len(groups[keys[k]])) for k in pick])
        vals = [roc_auc_score(d[label][sel] > t, d[s][sel]) for d, t in zip(ds, taus)
                if 0 < (d[label][sel] > t).sum() < len(sel)]
        if vals:
            bs.append(np.mean(vals))
    return np.percentile(bs, [2.5, 97.5])


def main(paths, data="data/cow_v1.h5"):
    topos = load_topologies(data)
    ds = [np.load(p) for p in paths]
    d0 = ds[0]
    ood = d0["idx_test_ood"]
    subsets = {"all": ood, "physiological": ood[d0["physiological"][ood].astype(bool)]}
    rng = np.random.default_rng(0)
    print(" + ".join(paths))
    for sub, idx in subsets.items():
        groups = {}
        for i in idx:
            groups.setdefault(topos[i], []).append(i)
        keys = list(groups)
        for label in ("err_Q", "err_A"):
            print(f"\n[{sub}] n={len(idx)} topologies={len(keys)} label {label}")
            print(f"{'rule':11s} {'bad%':>5s} " + " ".join(f"{s:>6s}" for s in SHORT))
            for name, rule in RULES:
                taus = [rule(d[label][d["idx_val"]]) if name != "OOD median" else np.median(d[label][idx]) for d in ds]
                bad = np.mean([(d[label][idx] > t).mean() for d, t in zip(ds, taus)])
                row = [np.mean([roc_auc_score(d[label][idx] > t, d[s][idx]) for d, t in zip(ds, taus)]) for s in SIGS]
                print(f"{name:11s} {100 * bad:5.1f} " + " ".join(f"{a:6.2f}" for a in row))
            taus = [np.percentile(d[label][d["idx_val"]], 90) for d in ds]
            cis = [cluster_ci(ds, taus, label, s, groups, keys, rng) for s in SIGS]
            print("p90 topology-cluster 95% CI: " + "  ".join(f"{n} [{lo:.2f}, {hi:.2f}]" for n, (lo, hi) in zip(SHORT, cis)))


if __name__ == "__main__":
    main(sys.argv[1:])
