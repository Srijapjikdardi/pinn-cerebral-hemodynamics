"""Evaluate the pre-registered hypotheses H1-H7 of docs/preregistration_b37.md from the saved npz files.
Usage: NETWORK_CONFIG=configs/network_boileau37.yaml RELIAB_DATA=data/b37_v1.h5 PYTHONPATH=. python reliability/prereg_b37.py
Files: outputs/b37_{graph,projgraph,mlp}_{variant,complexity}.npz (+ _pred.npz). Nothing here is tuned; thresholds are those written
in the pre-registration. Labels: err_Q above val p90 (also reports the balanced OOD-median label for H5)."""
import json
import os
import sys

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

from reliability.goodhart import decompose

DATA = os.environ.get("RELIAB_DATA", "data/b37_v1.h5")
OUT = "outputs"


def meta_topos():
    with h5py.File(DATA, "r") as h5:
        meta = [json.loads(h5[k].attrs["meta"]) for k in sorted(h5.keys())]
    return meta, [tuple(sorted(m["absent"])) for m in meta]


def auc(y, x):
    return roc_auc_score(y, x) if 0 < y.sum() < len(y) else float("nan")


def cluster_ci(d, topos, key, label_key="err_Q", nboot=1000, seed=0, round_to=None):
    idx = d["idx_test_ood"]
    x_all = d[key]
    if round_to:
        x_all = np.round(x_all / round_to) * round_to
    tau = np.percentile(d[label_key][d["idx_val"]], 90)
    groups = {}
    for i in idx:
        groups.setdefault(topos[i], []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    point = auc(d[label_key][idx] > tau, x_all[idx])
    bs = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([rng.choice(groups[keys[k]], len(groups[keys[k]])) for k in pick])
        y = d[label_key][sel] > tau
        if 0 < y.sum() < len(y):
            bs.append(roc_auc_score(y, x_all[sel]))
    return point, *np.percentile(bs, [2.5, 97.5])


def s7_of(path, meta):
    d = np.load(path)
    pr = np.load(path.replace(".npz", "_pred.npz"))
    pos = {int(i): k for k, i in enumerate(pr["pred_idx"])}
    ood = d["idx_test_ood"]
    return np.array([decompose(pr["pred_mean"][pos[int(i)]].astype(np.float64), meta[i])["S7_meanflow"] for i in ood])


def main():
    meta, topos = meta_topos()
    res = {}
    for sp in ("variant", "complexity"):
        g = np.load(f"{OUT}/b37_graph_{sp}.npz")
        p = np.load(f"{OUT}/b37_projgraph_{sp}.npz")
        m = np.load(f"{OUT}/b37_mlp_{sp}.npz")
        ood = g["idx_test_ood"]
        r = {}
        sig = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch", "S5_ensemble_spread", "S6_input_ood"]
        r["err_ood"] = dict(graph=float(g["err_Q"][ood].mean()), proj=float(p["err_Q"][ood].mean()), mlp=float(m["err_Q"][ood].mean()))
        r["err_id"] = dict(graph=float(g["err_Q"][g["idx_test_id"]].mean()), mlp=float(m["err_Q"][m["idx_test_id"]].mean()))
        r["bad_share"] = {k: float((d["err_Q"][ood] > np.percentile(d["err_Q"][d["idx_val"]], 90)).mean()) for k, d in (("graph", g), ("proj", p), ("mlp", m))}
        r["H1"] = cluster_ci(g, topos, "S1_junction_mass")
        r["H2"] = cluster_ci(p, topos, "S1_junction_mass", round_to=1e-7)[0]
        s7g, s7p = s7_of(f"{OUT}/b37_graph_{sp}.npz", meta), s7_of(f"{OUT}/b37_projgraph_{sp}.npz", meta)
        r["H3_ratio"] = float(np.median(s7p) / np.median(s7g))
        yg = g["err_Q"][ood] > np.percentile(g["err_Q"][g["idx_val"]], 90)
        yp = p["err_Q"][ood] > np.percentile(p["err_Q"][p["idx_val"]], 90)
        r["H4"] = (auc(yg, s7g), auc(yp, s7p))
        r["H5"] = dict(p90=auc(yp, s7p), balanced=auc(p["err_Q"][ood] > np.median(p["err_Q"][ood]), s7p),
                       balanced_unproj=auc(g["err_Q"][ood] > np.median(g["err_Q"][ood]), s7g))
        order = np.argsort(-g["S1_junction_mass"][ood])
        k = int(round(0.3 * len(ood)))
        kept = np.delete(g["err_Q"][ood], order[:k]).mean()
        r["H6"] = dict(s1_gate_kept=float(kept), random_kept=float(g["err_Q"][ood].mean()))
        r["all_signals_graph"] = {s: float(auc(yg, g[s][ood])) for s in sig}
        r["all_signals_mlp"] = {s: float(auc(m["err_Q"][ood] > np.percentile(m["err_Q"][m["idx_val"]], 90), m[s][ood])) for s in sig}
        res[sp] = r
    json.dump(res, open(f"{OUT}/b37_prereg_results.json", "w"), indent=1, default=float)
    for sp, r in res.items():
        print(f"\n=== {sp}")
        print("OOD err_Q graph / projected / MLP: %.3f / %.3f / %.3f ; ID graph / MLP: %.3f / %.3f ; bad share %s" % (
            r["err_ood"]["graph"], r["err_ood"]["proj"], r["err_ood"]["mlp"], r["err_id"]["graph"], r["err_id"]["mlp"],
            {k: round(v, 2) for k, v in r["bad_share"].items()}))
        print("H1 S1 AUROC (graph) %.2f [%.2f, %.2f] -> %s" % (*r["H1"], "PASS" if r["H1"][0] >= 0.70 and r["H1"][1] > 0.5 else "FAIL"))
        print("H2 projected S1 rounded 1e-7: %.2f -> %s" % (r["H2"], "PASS" if abs(r["H2"] - 0.5) <= 0.1 else "FAIL"))
        print("H3 S7 median ratio %.1fx -> %s" % (r["H3_ratio"], "PASS" if r["H3_ratio"] >= 3 else "FAIL"))
        print("H4 S7 AUROC unproj %.2f proj %.2f -> %s" % (*r["H4"], "PASS" if r["H4"][1] > r["H4"][0] else "FAIL"))
        print("H5 projected S7 AUROC p90 %.2f (balanced %.2f; unprojected balanced %.2f) -> %s" % (
            r["H5"]["p90"], r["H5"]["balanced"], r["H5"]["balanced_unproj"], "PASS" if r["H5"]["p90"] >= 0.70 else "FAIL"))
        print("H6 mean err_Q of kept cases, S1 gate vs random at 30%%: %.4f vs %.4f -> %s" % (
            r["H6"]["s1_gate_kept"], r["H6"]["random_kept"], "PASS" if r["H6"]["s1_gate_kept"] < r["H6"]["random_kept"] else "FAIL"))
        print("H7 graph < MLP OOD err_Q: %.3f vs %.3f -> %s" % (r["err_ood"]["graph"], r["err_ood"]["mlp"], "PASS" if r["err_ood"]["graph"] < r["err_ood"]["mlp"] else "FAIL"))
        print("all signals AUROC graph:", {k[:2]: round(v, 2) for k, v in r["all_signals_graph"].items()})
        print("all signals AUROC mlp  :", {k[:2]: round(v, 2) for k, v in r["all_signals_mlp"].items()})


if __name__ == "__main__":
    main()
