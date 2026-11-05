"""Solver-fallback policy evaluation. Usage: python reliability/fallback.py outputs/rel_variant.npz [data/cow_v1.h5]
Policy: surrogate always runs; if score > threshold the verified solver is called and its result replaces the
surrogate output. The dataset's solver output is the ground truth, so a deferred case has error 0 by construction
(this evaluates the policy, it is not an independent accuracy check of the solver).
System error = mean over cases of (0 if deferred else err_Q). Time/case = t_sur + t_sig + defer_rate * t_solver.
Thresholds are calibrated on validation only (in-distribution): defer if score > val quantile q.
Fused score: logistic regression on standardised log-signals (S1,S3,S4,S5,S6), fit on val with 'bad' = err_Q > val p90."""
import json
import sys
import time

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

SIG = ["S1_junction_mass", "S3_pde_residual", "S4_inlet_mismatch", "S5_ensemble_spread", "S6_input_ood"]


def surrogate_latency(n_members=5, reps=200):
    m = torch.nn.Sequential(*[l for _ in range(4) for l in (torch.nn.Linear(256, 256), torch.nn.GELU())],
                            torch.nn.Linear(256, 15 * 2 * 80 * 21))
    x = torch.randn(1, 256)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t = time.perf_counter()
        for _ in range(reps):
            m(x)
    return n_members * (time.perf_counter() - t) / reps


def curve(score, err, fracs):
    o = np.argsort(-score)
    out = []
    for f in fracs:
        k = int(round(f * len(err)))
        mask = np.ones(len(err), bool)
        mask[o[:k]] = False
        out.append(err[mask].sum() / len(err))
    return np.array(out)


def main(path, h5="data/cow_v1.h5", t_sig=0.004):
    d = np.load(path)
    err = d["err_Q"]
    with h5py.File(h5, "r") as f:
        wall = np.array([json.loads(f[k].attrs["meta"])["wall_s"] for k in sorted(f.keys())])
    t_solver = float(wall.mean())
    t_sur = surrogate_latency()
    print(f"{path}\nt_solver mean {t_solver:.2f} s/case (dataset wall_s, 2 workers in parallel), t_surrogate {t_sur*1e3:.2f} ms "
          f"(5 members, batch 1), t_signals {t_sig*1e3:.1f} ms; speed ratio solver/(surrogate+signals) = {t_solver/(t_sur+t_sig):.0f}x")
    val, tid, tood = d["idx_val"], d["idx_test_id"], d["idx_test_ood"]
    L = np.log(np.stack([d[s] for s in SIG], 1) + 1e-12)
    mu, sd = L[val].mean(0), L[val].std(0)
    Z = (L - mu) / sd
    tau = np.percentile(err[val], 90)
    clf = LogisticRegression(max_iter=1000).fit(Z[val], (err[val] > tau).astype(int))
    scores = {"S1_junction_mass": d["S1_junction_mass"], "S4_inlet_mismatch": d["S4_inlet_mismatch"],
              "S3_pde_residual": d["S3_pde_residual"], "S5_ensemble_spread": d["S5_ensemble_spread"],
              "FUSED(val-fit)": clf.decision_function(Z)}
    fr = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    for name, idx in (("test_id", tid), ("test_ood", tood)):
        e = err[idx]
        print(f"\n[{name}] n={len(idx)}  surrogate-only mean err {e.mean():.3f}")
        print("system error (rows) vs defer fraction (columns, top-score cases sent to solver)")
        print(f"{'policy':22s}" + "".join(f"{f:>7.0%}" for f in fr))
        print(f"{'random':22s}" + "".join(f"{(1-f)*e.mean():7.3f}" for f in fr))
        print(f"{'oracle':22s}" + "".join(f"{v:7.3f}" for v in curve(e, e, fr)))
        for s, sc in scores.items():
            print(f"{s:22s}" + "".join(f"{v:7.3f}" for v in curve(sc[idx], e, fr)))
    print("\nVal-calibrated operating points (defer if score > val quantile; threshold never sees OOD):")
    print(f"{'policy':20s} {'q':>5s} | {'ID defer':>8s} {'ID err':>7s} {'ID t/case':>9s} | {'OOD defer':>9s} {'OOD err':>8s} {'OOD t/case':>10s} {'err reduction vs surrogate':>27s}")
    for s in ("S1_junction_mass", "S4_inlet_mismatch", "FUSED(val-fit)"):
        for q in (0.80, 0.90, 0.95):
            thr = np.quantile(scores[s][val], q)
            row = []
            for idx in (tid, tood):
                dm = scores[s][idx] > thr
                se = np.where(dm, 0.0, err[idx]).mean()
                tt = t_sur + t_sig + dm.mean() * t_solver
                row.append((dm.mean(), se, tt, err[idx].mean()))
            print(f"{s:20s} {q:5.2f} | {row[0][0]:8.2f} {row[0][1]:7.3f} {row[0][2]:9.2f} | {row[1][0]:9.2f} {row[1][1]:8.3f} {row[1][2]:10.2f} "
                  f"{1-row[1][1]/row[1][3]:15.0%} (random at same rate {row[1][0]:.0%})")
    print(f"always-solver time/case {t_solver:.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
