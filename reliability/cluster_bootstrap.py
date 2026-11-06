"""AUROC within held-out topologies with a topology-cluster bootstrap (resample topologies, then samples within each), which
respects the correlation between samples of the same topology. Usage: python reliability/cluster_bootstrap.py outputs/rel_variant.npz [more.npz ...]
Pools all given files (independent ensembles) by averaging their signal ranks is NOT done; each file is bootstrapped with the same
resampling draws and the mean AUROC over files is reported with its percentile CI."""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from data.splits import load_topologies

SIGS = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch",
        "S5_ensemble_spread", "S6_input_ood"]


def main(paths, data="data/cow_v1.h5", nboot=1000, seed=0):
    topos = load_topologies(data)
    ds = [np.load(p) for p in paths]
    idx = ds[0]["idx_test_ood"]
    groups = {}
    for i in idx:
        groups.setdefault(topos[i], []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    for label in ("err_Q", "err_A"):
        taus = [np.percentile(d[label][d["idx_val"]], 90) for d in ds]
        print(f"\nlabel {label}: {len(keys)} held-out topologies, {len(idx)} samples, files {len(ds)}")
        print(f"{'signal':24s} {'AUROC':>6s}  topology-cluster 95% CI")
        for s in SIGS:
            point = np.mean([roc_auc_score(d[label][idx] > t, d[s][idx]) for d, t in zip(ds, taus)])
            bs = []
            for _ in range(nboot):
                pick = rng.integers(0, len(keys), len(keys))
                sel = np.concatenate([rng.choice(groups[keys[k]], len(groups[keys[k]])) for k in pick])
                vals = []
                for d, t in zip(ds, taus):
                    y = d[label][sel] > t
                    if 0 < y.sum() < len(y):
                        vals.append(roc_auc_score(y, d[s][sel]))
                if vals:
                    bs.append(np.mean(vals))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{s:24s} {point:6.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
