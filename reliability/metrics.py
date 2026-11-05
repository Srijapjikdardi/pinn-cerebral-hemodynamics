"""Evaluate reliability signals against true surrogate error. Usage: python reliability/metrics.py outputs/rel_variant.npz
Label 'bad' = err_Q above the 90th percentile of validation (in-distribution) err_Q.
Reports per subset: AUROC, Spearman, AURC (mean error of retained cases over coverages 10..100%) vs random and oracle.
Fused score: logistic regression on rank-normalised signals, fit on val only (no OOD labels used)."""
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SIGS = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch",
        "S5_ensemble_spread", "S6_input_ood"]


def aurc(score, err, covs=np.linspace(0.1, 1.0, 10)):
    o = np.argsort(score)
    e = err[o]
    return float(np.mean([e[:max(1, int(round(c * len(e))))].mean() for c in covs]))


def evaluate(path):
    d = np.load(path)
    err = d["err_Q"]
    val = d["idx_val"]
    tau = np.percentile(err[val], 90)
    rank = {s: rankdata(d[s]) / len(err) for s in SIGS}
    Z = np.stack([rank[s] for s in SIGS], 1)
    lab_val = (err[val] > tau).astype(int)
    fuse = LogisticRegression(C=1.0, max_iter=1000).fit(Z[val], lab_val)
    fused = fuse.decision_function(Z)
    scores = {s: d[s] for s in SIGS}
    scores["FUSED(val-fit)"] = fused
    print(f"{path}: tau(err_Q p90 of val)={tau:.3f}; n_val={len(val)}")
    subsets = {"test_id": d["idx_test_id"], "test_ood": d["idx_test_ood"]}
    subsets["pooled"] = np.concatenate([subsets["test_id"], subsets["test_ood"]])
    for name, idx in subsets.items():
        if len(idx) == 0:
            continue
        e = err[idx]
        y = (e > tau).astype(int)
        print(f"\n[{name}] n={len(idx)} bad={y.mean():.2f} mean err {e.mean():.3f}  "
              f"AURC random {e.mean():.3f} oracle {aurc(e, e):.3f}")
        print(f"{'signal':24s} {'AUROC':>6s} {'Spearman':>8s} {'AURC':>6s}")
        for s, sc in scores.items():
            x = sc[idx]
            au = roc_auc_score(y, x) if 0 < y.sum() < len(y) else float("nan")
            print(f"{s:24s} {au:6.3f} {spearmanr(x, e)[0]:8.3f} {aurc(x, e):6.3f}")
    print("\nsignal floor on exact solver output (median) vs predicted (median):")
    for s in SIGS[:4]:
        print(f"  {s:22s} truth {np.median(d['truth_' + s]):.4g}  pred {np.median(d[s]):.4g}")


if __name__ == "__main__":
    evaluate(sys.argv[1])
