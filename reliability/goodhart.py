"""Where does the inconsistency go when junction mass is enforced? Decomposes signals on saved ensemble-mean predictions.

For each surrogate npz (from training/reliability_eval.py, must contain pred_idx / pred_mean) computes per sample:
  S1_end      : junction mass at the vessel-end grid points (the points the projection layer corrects; = S1)
  S1_inner    : junction mass using the first interior grid point of each incident vessel (j = 1 and nx-2)
  S3_mass     : interior continuity residual A_t + Q_x (RMS, normalised as in signals.py)
  S3_mom      : interior momentum residual (RMS, normalised)
  S7_meanflow : periodic-state mass balance per vessel: spread along x of the cycle-mean flow, relative to the vessel's
                cycle-mean flow scale. Exact solution has a constant cycle-mean Q along each vessel (A periodic).
Caveat: predictions are stored as float16 (relative precision ~1e-3), so S1_end/S1_inner of a junction-projected surrogate
read the storage quantisation (~2e-3), not its exact-to-round-off S1 (1e-8, in the main npz). S3/S7 values are 10-100x larger.
and AUROC within held-out topologies against the err_Q / err_A "bad" labels (val 90th percentile).

Usage: PYTHONPATH=. python reliability/goodhart.py outputs/rel_variant.npz outputs/relproj_variant.npz ...
"""
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from data.cow_dataset import VESSELS, load
from reliability.signals import KR, NODES, RHO, _fields

DATA = "data/cow_v1.h5"


def decompose(pred, m):
    nt, nx = pred.shape[2], pred.shape[3]
    pres = set(m["vessels"])
    F = {n: _fields(pred, n, m, nx) for n in VESSELS if n in pres}
    qscale = max(np.abs(f[1]).max() for f in F.values()) + 1e-30
    s_end, s_in = [], []
    for node in NODES.values():
        ends = [(v, e) for v, e in node if v in pres]
        if len(ends) < 2:
            continue
        qe, qi = [], []
        for v, e in ends:
            Q = F[v][1]
            sg = 1.0 if e == 1 else -1.0
            qe.append(sg * Q[:, nx - 1 if e == 1 else 0])
            qi.append(sg * Q[:, nx - 2 if e == 1 else 1])
        s_end.append(np.abs(np.sum(qe, 0)).mean() / qscale)
        s_in.append(np.abs(np.sum(qi, 0)).mean() / qscale)
    dt = m["period"] / nt
    rm, rq, mf = [], [], []
    for v, (A, Q, P, c, h) in F.items():
        At = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * dt)
        Qt = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0)) / (2 * dt)
        Qx = np.gradient(Q, h, axis=1)
        k3 = c["b"] / (3 * RHO * c["a0"])
        fx = np.gradient(Q ** 2 / A + k3 * A ** 1.5, h, axis=1)
        mass, mom = At + Qx, Qt + fx + KR * Q / A
        sA = np.abs(At).mean() + np.abs(Qx).mean() + 1e-30
        sQ = np.abs(Qt).mean() + np.abs(fx).mean() + np.abs(KR * Q / A).mean() + 1e-30
        sl = slice(1, nx - 1)
        rm.append(math.sqrt((mass[:, sl] ** 2).mean()) / sA)
        rq.append(math.sqrt((mom[:, sl] ** 2).mean()) / sQ)
        qbar = Q.mean(0)
        mf.append((qbar.max() - qbar.min()) / (np.abs(Q).mean() + 1e-30))
    return dict(S1_end=np.mean(s_end), S1_inner=np.mean(s_in), S3_mass=np.mean(rm), S3_mom=np.mean(rq),
                S7_meanflow=np.mean(mf))


def labels(d, key):
    thr = np.quantile(d[key][d["idx_val"]], 0.9)
    return d[key] > thr


def main(paths):
    import h5py, json
    with h5py.File(DATA, "r") as h5:
        meta = [json.loads(h5[k].attrs["meta"]) for k in sorted(h5.keys())]
    keys = ["S1_end", "S1_inner", "S3_mass", "S3_mom", "S7_meanflow"]
    floor = None
    for p in paths:
        d = np.load(p)
        pp = Path(p.replace(".npz", "_pred.npz"))
        if not pp.exists():
            print(p, "has no", pp, "; rerun reliability_eval"); continue
        pr = np.load(pp)
        pidx, pred_mean = pr["pred_idx"], pr["pred_mean"]
        pos = {int(i): k for k, i in enumerate(pidx)}
        ood = d["idx_test_ood"]
        rows = [decompose(pred_mean[pos[int(i)]].astype(np.float64), meta[i]) for i in ood]
        if floor is None:
            sub = list(ood[:200])
            _, Ysub, _, msub = load(DATA, indices=sub)
            tr = [decompose(Ysub[j].astype(np.float64), msub[j]) for j in range(len(sub))]
            floor = {k: float(np.median([r[k] for r in tr])) for k in keys}
            print("exact-solver floor (median):", {k: f"{v:.2e}" for k, v in floor.items()})
        np.savez(p.replace(".npz", "_decomp.npz"), idx=ood, floor=np.array([floor[k] for k in keys]), keys=np.array(keys),
                 **{k: np.array([r[k] for r in rows]) for k in keys})  # small cache for figures
        print(f"\n{p}  OOD err_Q {d['err_Q'][ood].mean():.3f}")
        for k in keys:
            s = np.array([r[k] for r in rows])
            aq = roc_auc_score(labels(d, "err_Q")[ood], s)
            aa = roc_auc_score(labels(d, "err_A")[ood], s)
            print(f"  {k:12s} median {np.median(s):.2e} ({np.median(s) / floor[k]:7.1f}x floor)  AUROC Q {aq:.2f}  A {aa:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
