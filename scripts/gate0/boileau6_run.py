"""Boileau et al. 2015 benchmark 6 (ADAN56, 56 largest systemic arteries, 61 segments split into 77 pieces) with physics/solver_1d.py,
compared with the six reference 1D codes at the 12 named sites (segment midpoints).

Data: the paper's supplement Benchmark6_ADAN56 (mesh.txt, inflow.txt, NumData) copied to data/adan56/ (gitignored, third-party).
General parameters (paper Table XI; values taken from Yanbarisov et al. 2023, Table 4, which reproduces them, because Table XI could
not be read from the open-access PDF here; the wall-thickness law and Pd were read from the paper's section 4.6 text):
  rho 1040 kg/m3, mu 4.0 mPa s, velocity profile zeta = 2 (friction 8 pi mu / rho, our default), E = 225 kPa everywhere,
  Pd = P0 = 10 kPa at Ad = A0, Pext = 0, Pout = 0,
  h = R0 [a exp(b R0) + c exp(d R0)], a 0.2802, b -5.053 /cm, c 0.1324, d -0.1114 /cm (R0 in cm).
Terminals: three-element Windkessel R1, R2, C from mesh.txt (cgs -> SI: R x 1e5, C x 1e-5). Inflow: inflow.txt (period 1.0 s, linear).
Tapered segments are chains of uniform sub-vessels (<= --sub cm) with linearly varying radius and the local h.
Usage: python scripts/gate0/boileau6_run.py [--dx 2e-3 --sub 1.0]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.solver_1d import build_network, simulate_periodic  # noqa: E402

D = Path(__file__).resolve().parents[2] / "data" / "adan56"
RHO, MU, E, PD, T = 1040.0, 4.0e-3, 2.25e5, 1.0e4, 1.0
CODES = ["DCG", "LCG", "FEM", "FVM", "McC", "STM"]
SITES = [("AorticArchI", "1"), ("ThoAortaIII", "29"), ("AbdomAortaV", "47"), ("RightCCA", "5"), ("RightRenal", "44"),
         ("RightCommonIliac", "48"), ("RightICA", "13"), ("RightRadial", "8"), ("RightInternalIliac", "51"),
         ("RightPostInteross", "10b"), ("RightFemoralII", "53a"), ("RightAntTibial", "54")]


def wall_h(r_cm):
    return r_cm * (0.2802 * math.exp(-5.053 * r_cm) + 0.1324 * math.exp(-0.1114 * r_cm)) * 1e-2  # m


def build(dx, sub_cm, n_tab=1000):
    rows = [l.split() for l in open(D / "mesh.txt") if l.strip()]
    vessels, nodes, outlets, chains = [], {}, {}, {}
    for r in rows:
        sid, na, nb = r[0], int(r[1]), int(r[2])
        L, ra, rb, R1, R2, C = (float(x) for x in r[3:9])
        n = max(1, int(math.ceil(L / sub_cm)))
        names = []
        for k in range(n):
            rm = ra + (rb - ra) * (k + 0.5) / n
            nm = f"s{sid}_{k}"
            vessels.append(dict(name=nm, L=L / n * 1e-2, r0=rm * 1e-2, h=wall_h(rm), E=E))
            names.append(nm)
        chains[sid] = names
        nodes.setdefault(na, []).append((names[0], 0))
        nodes.setdefault(nb, []).append((names[-1], 1))
        for k in range(n - 1):
            nodes[f"{sid}_{k}"] = [(names[k], 1), (names[k + 1], 0)]
        if R1 > 0:
            outlets[names[-1]] = dict(R1=R1 * 1e5, R2=R2 * 1e5, C=C * 1e-5)
    inlet = nodes.pop(0)
    assert len(inlet) == 1
    nl = []
    for key, ends in nodes.items():
        if len(ends) == 1:
            assert ends[0][0] in outlets, f"dangling end {ends} at {key}"
            continue
        nl.append(ends)
    tab = np.loadtxt(D / "inflow.txt")
    t = np.arange(n_tab) * T / n_tab
    wtab = np.interp(t, tab[:, 0], tab[:, 1]) * 1e-6
    net = build_network(vessels, nl, {inlet[0][0]: 1.0}, outlets, rho=RHO, mu=MU, sigma=0.5, pv=0.0 - PD,
                        wtab=wtab, period=T, dx_target=dx)
    return net, chains


def mid(out, chain):
    v = chain[len(chain) // 2]
    a = out[v]
    if len(chain) % 2:
        j = a["A"].shape[1] // 2
    else:  # even number of pieces: the segment midpoint is the left end of the middle piece
        j = 0
    return (PD + a["P"][:, j]) * 1e-3, a["Q"][:, j] * 1e6


def load_ref(code, n):
    out = {}
    for q in ("P", "Q"):
        x = np.loadtxt(D / "NumData" / f"{code}_{q}.txt")
        t = 9.0 + (np.arange(n) + 1) * T / n  # our sample i is at phase (i+1)/n of the last cycle
        out[q] = np.stack([np.interp(t, x[:, 0], x[:, c + 1]) for c in range(len(SITES))], 1)
    return out


def rrms(a, b):
    return 100 * np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))


def amp(a, b):  # relative to the pulse amplitude of the reference (pressure is offset-dominated)
    return 100 * np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=2e-3)
    ap.add_argument("--sub", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="outputs/boileau6.npz")
    ap.add_argument("--reuse", action="store_true", help="re-report a saved run without solving")
    a = ap.parse_args()
    if a.reuse and Path(a.out).exists():
        o = np.load(a.out); P, Q = o["P"], o["Q"]
        refs = {c: load_ref(c, a.n) for c in CODES}
        return report(P, Q, refs)
    net, chains = build(a.dx, a.sub)
    print("sub-vessels", len(net.names), "cells", int(net.ncell.sum()), flush=True)
    out = simulate_periodic(net, 0.0, n_out=a.n, tol=1e-6, max_cycles=60, progress=lambda c, ch: print(f"cycle {c} change {ch:.1e}", flush=True))
    dg = out["diagnostics"]
    print("diagnostics", {k: dg[k] for k in dg if not isinstance(dg[k], (list, np.ndarray))})
    P = np.stack([mid(out, chains[s])[0] for _, s in SITES], 1)
    Q = np.stack([mid(out, chains[s])[1] for _, s in SITES], 1)
    refs = {c: load_ref(c, a.n) for c in CODES}
    np.savez(a.out, P=P, Q=Q, sites=[s for s, _ in SITES])
    report(P, Q, refs)


def report(P, Q, refs):
    print(f"\n{'site':20s} {'P ours/6-mean':>13s} {'P spread':>9s} {'P amp-norm':>10s} {'Q ours/6-mean':>13s} {'Q spread':>9s}   (relative RMS %)")
    # Data check independent of our solution: a reference series is dropped at a site if its relative RMS distance to the median of
    # the other five codes exceeds 20% (the DCG flow file contains spikes, e.g. max 6540 ml/s in the right CCA vs 30.6 for the others).
    def keep(q, k):
        ok = []
        for c in CODES:
            others = np.median([refs[o][q][:, k] for o in CODES if o != c], 0)
            if rrms(refs[c][q][:, k], others) <= 20.0:
                ok.append(c)
            else:
                dropped.add((c, q, name))
        return ok
    dropped = set()
    rows = []
    for k, (name, _) in enumerate(SITES):
        cP, cQ = keep("P", k), keep("Q", k)
        mP = np.mean([refs[c]["P"][:, k] for c in cP], 0)
        mQ = np.mean([refs[c]["Q"][:, k] for c in cQ], 0)
        sP = np.mean([rrms(refs[c]["P"][:, k], mP) for c in cP])
        sQ = np.mean([rrms(refs[c]["Q"][:, k], mQ) for c in cQ])
        r = (rrms(P[:, k], mP), sP, amp(P[:, k], mP), rrms(Q[:, k], mQ), sQ)
        rows.append(r)
        print(f"{name:20s} {r[0]:13.2f} {r[1]:9.2f} {r[2]:10.2f} {r[3]:13.2f} {r[4]:9.2f}")
    rows = np.array(rows)
    print(f"{'median':20s} {np.median(rows[:, 0]):13.2f} {np.median(rows[:, 1]):9.2f} {np.median(rows[:, 2]):10.2f} "
          f"{np.median(rows[:, 3]):13.2f} {np.median(rows[:, 4]):9.2f}")
    print("dropped reference series (code, quantity, site):", sorted(dropped))


if __name__ == "__main__":
    main()
