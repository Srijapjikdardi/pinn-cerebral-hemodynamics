"""Paper figures from outputs/*.npz (no training). Usage: PYTHONPATH=. python scripts/make_figures.py [--out figures]
Fig 1  AUROC per signal with topology-cluster 95% CI, MLP vs graph, variant and complexity (flow-error label).
Fig 2  Risk-coverage on held-out topologies (variant): mean flow error of retained cases vs coverage.
Fig 3  Goodhart: S1 and S7 AUROC without / with the junction-projection layer, four model x split pairs.
Fig 4  Flow error vs number of absent vessels, MLP vs graph (test_id + test_ood of the complexity split).
Fig 5  Leave-one-vessel-out: OOD flow error and S1/S7 AUROC per held-out vessel (if outputs/relgraph_vessel_*.npz exist).
Colors: validated categorical slots (dataviz reference palette, light mode); identity also carried by marker shape and labels.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data.splits import load_topologies  # noqa: E402

C = {"mlp": "#2a78d6", "graph": "#eb6834", "random": "#8a8a85", "oracle": "#1f1f1e", "proj": "#1baf7a"}
MK = {"mlp": "o", "graph": "s"}
SIGS = ["S1_junction_mass", "S2_junction_pressure", "S3_pde_residual", "S4_inlet_mismatch", "S5_ensemble_spread", "S6_input_ood"]
NAMES = ["S1\njunction\nmass", "S2\njunction\npressure", "S3\nPDE\nresidual", "S4\ninlet\nflow", "S5\nensemble\nspread", "S6\ninput\ndistance"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#1f1f1e", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "savefig.dpi": 300,
                     "legend.frameon": False})


def label(d, key="err_Q"):
    return d[key] > np.percentile(d[key][d["idx_val"]], 90)


def cluster_auc(files, sig, topos, key="err_Q", nboot=400, seed=0):
    ds = [np.load(f) for f in files]
    idx = ds[0]["idx_test_ood"]
    groups = {}
    for i in idx:
        groups.setdefault(topos[i], []).append(i)
    keys = list(groups)
    ys = [label(d, key) for d in ds]
    point = np.mean([roc_auc_score(y[idx], d[sig][idx]) for d, y in zip(ds, ys)])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        sel = np.concatenate([rng.choice(groups[keys[k]], len(groups[keys[k]])) for k in rng.integers(0, len(keys), len(keys))])
        v = [roc_auc_score(y[sel], d[sig][sel]) for d, y in zip(ds, ys) if 0 < y[sel].sum() < len(sel)]
        if v:
            bs.append(np.mean(v))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, lo, hi


def fig1(out, topos):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, sp in zip(axes, ("variant", "complexity")):
        for k, (m, files) in enumerate((("mlp", [f"outputs/rel_{sp}.npz", f"outputs/rel_{sp}_s200.npz"]),
                                        ("graph", [f"outputs/relgraph_{sp}.npz", f"outputs/relgraph_{sp}_s700.npz"]))):
            files = [f for f in files if Path(f).exists()]
            r = np.array([cluster_auc(files, s, topos) for s in SIGS])
            x = np.arange(len(SIGS)) + (k - 0.5) * 0.28
            ax.errorbar(x, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt=MK[m], ms=5, color=C[m], elinewidth=1.4,
                        capsize=0, label={"mlp": "MLP", "graph": "Vessel-graph"}[m])
        ax.axhline(0.5, color="#8a8a85", lw=1, ls="--")
        ax.text(-0.45, 0.505, "chance", ha="left", va="bottom", fontsize=7.5, color="#4a4a46")
        ax.set_xticks(range(len(SIGS)), NAMES, fontsize=7.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"`{sp}` split: held-out topologies", fontsize=9)
    axes[0].set_ylabel("AUROC for large flow error\n(topology-cluster 95% CI)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_auroc_signals.png"); fig.savefig(out / "fig1_auroc_signals.pdf")


def rc_curve(err, score):
    o = np.argsort(score)  # keep lowest-score cases first
    cov = np.linspace(0.1, 1.0, 19)
    return cov, np.array([err[o[:max(1, int(round(c * len(err))))]].mean() for c in cov])


def fig2(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (m, f) in zip(axes, (("mlp", "outputs/rel_variant.npz"), ("graph", "outputs/relgraph_variant.npz"))):
        d = np.load(f)
        o = d["idx_test_ood"]
        e = d["err_Q"][o]
        cov = np.linspace(0.1, 1.0, 19)
        ax.plot(cov, np.full_like(cov, e.mean()), color=C["random"], lw=2, ls="--", label="random deferral")
        c, v = rc_curve(e, e); ax.plot(c, v, color=C["oracle"], lw=2, ls=":", label="oracle")
        for s, col, nm in (("S1_junction_mass", C[m], "S1 junction mass"), ("S5_ensemble_spread", C["proj"], "S5 ensemble spread")):
            c, v = rc_curve(e, d[s][o]); ax.plot(c, v, color=col, lw=2, label=nm)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.set_xlabel("coverage (fraction kept by the surrogate)")
        ax.set_title({"mlp": "MLP", "graph": "Vessel-graph"}[m] + ", `variant` held-out topologies", fontsize=9)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("mean flow rel. L2 error of kept cases")
    fig.tight_layout()
    fig.savefig(out / "fig2_risk_coverage.png"); fig.savefig(out / "fig2_risk_coverage.pdf")


def fig3(out):
    pairs = [("MLP", "variant", "rel_variant", "relproj_variant"), ("Graph", "variant", "relgraph_variant", "relprojgraph_variant"),
             ("MLP", "complexity", "rel_complexity", "relproj_complexity"),
             ("Graph", "complexity", "relgraph_complexity", "relprojgraph_complexity")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, sig in zip(axes, ("S1", "S7")):
        for k, (m, sp, a, b) in enumerate(pairs):
            vals = []
            for stem in (a, b):
                d = np.load(f"outputs/{stem}.npz")
                if sig == "S1":
                    o = d["idx_test_ood"]; s = d["S1_junction_mass"][o]
                    s = np.round(s / 1e-7) if s.max() < 1e-6 else s  # below solver floor: round-off only (see audit)
                else:
                    dc = np.load(f"outputs/{stem}_decomp.npz"); o = dc["idx"]; s = dc["S7_meanflow"]
                vals.append(roc_auc_score(label(d)[o], s))
            col = C["mlp"] if m == "MLP" else C["graph"]
            ax.plot([0, 1], vals, color=col, lw=2, marker=MK[m.lower()], ms=5, ls="-" if sp == "variant" else "--")
            if sig == "S7":
                ax.text(1.04, vals[1], f"{m}, {sp}", va="center", fontsize=7.5)
        ax.axhline(0.5, color="#8a8a85", lw=1, ls=":")
        ax.set_xticks([0, 1], ["no constraint", "hard junction\nprojection"])
        ax.set_xlim(-0.15, 1.15) if sig == "S1" else ax.set_xlim(-0.15, 1.75)
        if sig == "S1":
            ax.text(1.0, 0.47, "all four 0.50\n(S1 at round-off,\nread at 1e-7)", ha="center", va="top", fontsize=7.5, color="#4a4a46")
        ax.set_title({"S1": "S1 junction mass (the enforced check)", "S7": "S7 mean-flow balance (where the error moves)"}[sig],
                     fontsize=9)
    axes[0].set_ylabel("AUROC for large flow error")
    axes[0].set_ylim(0.3, 1.0)
    for m, mk, col in (("MLP", "o", C["mlp"]), ("Vessel-graph", "s", C["graph"])):
        axes[0].plot([], [], color=col, marker=mk, lw=2, label=m)
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[1].plot([], [], color="#4a4a46", lw=1.5, ls="-", label="`variant` split")
    axes[1].plot([], [], color="#4a4a46", lw=1.5, ls="--", label="`complexity` split")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_goodhart.png"); fig.savefig(out / "fig3_goodhart.pdf")


def fig4(out):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for m, f in (("mlp", "outputs/rel_complexity.npz"), ("graph", "outputs/relgraph_complexity.npz")):
        d = np.load(f)
        idx = np.concatenate([d["idx_test_id"], d["idx_test_ood"]])
        na, e = d["n_absent"][idx], d["err_Q"][idx]
        ks = sorted(set(na))
        med = [np.median(e[na == k]) for k in ks]
        q1 = [np.percentile(e[na == k], 25) for k in ks]; q3 = [np.percentile(e[na == k], 75) for k in ks]
        ax.plot(ks, med, color=C[m], marker=MK[m], lw=2, ms=5, label={"mlp": "MLP", "graph": "Vessel-graph"}[m])
        ax.fill_between(ks, q1, q3, color=C[m], alpha=0.15, lw=0)
    ax.axvspan(3.5, 6.5, color="#e6e5e0", alpha=0.6, lw=0)
    ax.text(3.6, ax.get_ylim()[1] * 0.95, "never seen\nin training", va="top", fontsize=7.5, color="#4a4a46")
    ax.set_xlabel("absent variable vessels"); ax.set_ylabel("flow rel. L2 error (median, IQR)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_error_vs_absent.png"); fig.savefig(out / "fig4_error_vs_absent.pdf")


def fig5(out):
    vs = [v for v in ("ACoA", "PCoA_L", "A1_L", "P1_L") if Path(f"outputs/relgraph_vessel_{v}.npz").exists()
          and Path(f"outputs/relprojgraph_vessel_{v}_decomp.npz").exists()]
    if not vs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(vs))
    e = [np.load(f"outputs/relgraph_vessel_{v}.npz") for v in vs]
    axes[0].bar(x, [d["err_Q"][d["idx_test_ood"]].mean() for d in e], width=0.55, color=C["graph"])
    axes[0].bar(x, [d["err_Q"][d["idx_test_id"]].mean() for d in e], width=0.55, color="#f6c3ae")
    axes[0].set_xticks(x, vs); axes[0].set_ylabel("mean flow rel. L2 error")
    axes[0].set_title("Graph surrogate: held out (dark) vs seen (light)", fontsize=9)
    for sig, off, col, nm in (("S1", -0.15, C["graph"], "S1, no constraint"), ("S7", 0.15, C["proj"], "S7, projected")):
        vals = []
        for v in vs:
            if sig == "S1":
                d = np.load(f"outputs/relgraph_vessel_{v}.npz"); o = d["idx_test_ood"]
                vals.append(roc_auc_score(label(d)[o], d["S1_junction_mass"][o]))
            else:
                d = np.load(f"outputs/relprojgraph_vessel_{v}.npz"); dc = np.load(f"outputs/relprojgraph_vessel_{v}_decomp.npz")
                vals.append(roc_auc_score(label(d)[dc["idx"]], dc["S7_meanflow"]))
        axes[1].plot(x + off, vals, ls="none", marker="o" if sig == "S1" else "s", ms=6, color=col, label=nm)
    axes[1].axhline(0.7, color="#8a8a85", lw=1, ls="--"); axes[1].text(len(vs) - 0.6, 0.705, "pre-registered 0.70", fontsize=7, ha="right")
    axes[1].set_xticks(x, vs); axes[1].set_ylim(0.3, 1.0); axes[1].set_ylabel("AUROC, large flow error")
    axes[1].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_vessel_holdout.png"); fig.savefig(out / "fig5_vessel_holdout.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    topos = load_topologies("data/cow_v1.h5")
    fig1(out, topos); fig2(out); fig4(out)
    if Path("outputs/relprojgraph_complexity_decomp.npz").exists():
        fig3(out)
    fig5(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
