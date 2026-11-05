"""AUROC within held-out topologies, for Q-based and A-based labels, with bootstrap CIs (resampling test samples)
and spread across independent ensembles (different member seeds).
Usage: python reliability/robustness.py outputs/rel_variant.npz outputs/rel_variant_s200.npz ...
A-based label = err_A above the 90th percentile of validation err_A (P is a monotone function of A in this model)."""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

SIGS = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch",
        "S5_ensemble_spread", "S6_input_ood"]


def auc(y, x):
    return roc_auc_score(y, x) if 0 < y.sum() < len(y) else float("nan")


def one(path, label, rng, nboot=1000):
    d = np.load(path)
    err = d["err_Q"] if label == "Q" else d["err_A"]
    tau = np.percentile(err[d["idx_val"]], 90)
    idx = d["idx_test_ood"]
    y = (err[idx] > tau).astype(int)
    out = {}
    for s in SIGS:
        x = d[s][idx]
        a = auc(y, x)
        bs = []
        for _ in range(nboot):
            b = rng.integers(0, len(idx), len(idx))
            if 0 < y[b].sum() < len(b):
                bs.append(roc_auc_score(y[b], x[b]))
        out[s] = (a, *np.percentile(bs, [2.5, 97.5]))
    return out, y.mean()


def main(paths):
    rng = np.random.default_rng(0)
    for label in ("Q", "A"):
        res = [one(p, label, rng) for p in paths]
        print(f"\nlabel = {label}-error above val p90; AUROC within test_ood; first file bootstrap 95% CI; "
              f"mean +- sd over {len(paths)} ensembles; bad fraction {res[0][1]:.2f}")
        print(f"{'signal':24s} {'AUROC[CI] (file 1)':>24s} {'mean+-sd over ensembles':>26s}")
        for s in SIGS:
            a0, lo, hi = res[0][0][s]
            allv = np.array([r[0][s][0] for r in res])
            print(f"{s:24s} {a0:8.3f} [{lo:.3f},{hi:.3f}] {allv.mean():12.3f} +- {allv.std():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
