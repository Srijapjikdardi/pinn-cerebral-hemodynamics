"""Boileau et al. 2015 benchmark 5 (37-artery silicone network, in vitro data of Matthys et al. 2007) with physics/solver_1d.py.
Inputs (data/boileau37/, from the paper's supplementary zip): mesh.txt, inflow.txt. Fluid (Matthys 2007): rho 1050, mu 2.5 mPa s.
E from the supplied PWV per segment via the tube law consistent with the solver: c^2 = (2/3) E h / (rho r) (gives E = 1.2 MPa, the
value Matthys measured). Taper: each segment is a chain of uniform sub-vessels (<= --sub cm) with interpolated radius and PWV.
Terminals: single resistor R1 (mesh col 10, taken as x1e9 Pa s/m3, consistent with Matthys' total peripheral resistance) to p_out.
Data: copy mesh.txt and inflow.txt from the paper's supplement (Benchmark5_37ArteryNetwork.zip) into data/boileau37/ (gitignored,
third-party data, not redistributed here).
--mu-scale 2.53 emulates the paper's velocity-profile parameter zeta = 9: friction 2 pi mu zeta^2/(zeta-1) vs 8 pi mu (zeta = 2).
Usage: python scripts/gate0/boileau37_run.py --pout 0 --dx 2e-3 --sub 1.0 --out outputs/boileau37_pout0.npz
Saves one periodic cycle (827 samples, 1 ms) of P [kPa] and Q [ml/s] at the midpoint of every segment."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.solver_1d import build_network  # noqa: E402

D = Path(__file__).resolve().parents[2] / "data" / "boileau37"
RHO, MU = 1050.0, 2.5e-3
T = 0.827


def inflow_table(n=827):
    expr = [l for l in open(D / "inflow.txt").read().replace("\r", "").split("\n") if l.startswith("inflow=")][0][7:]
    PI = math.pi
    sin = np.sin
    t = np.arange(n) * T / n
    return 1e-6 * eval(expr, {"sin": sin, "PI": PI, "t": t, "T": T})  # ml/s -> m^3/s


def build(pout, dx, sub_cm, r1_scale=1e9, mu=MU):
    mesh = np.loadtxt(D / "mesh.txt")
    vessels, nodes, outlets = [], {}, {}
    chains = {}
    for row in mesh:
        sid, na, nb, L, r_a, r_b, h, c_a, c_b, r1 = row
        sid = int(sid)
        n = max(1, int(math.ceil(L / (sub_cm * 1e-2))))
        names = []
        for k in range(n):
            f0, f1 = k / n, (k + 1) / n
            fm = 0.5 * (f0 + f1)
            r = r_a + (r_b - r_a) * fm
            c = c_a + (c_b - c_a) * fm
            E = 1.5 * RHO * r * c ** 2 / h
            nm = f"s{sid}_{k}"
            vessels.append(dict(name=nm, L=L / n, r0=r, h=h, E=E))
            names.append(nm)
        chains[sid] = names
        nodes.setdefault(int(na), []).append((names[0], 0))
        nodes.setdefault(int(nb), []).append((names[-1], 1))
        for k in range(n - 1):
            nodes[f"{sid}_{k}"] = [(names[k], 1), (names[k + 1], 0)]
        if r1 > 0:
            outlets[names[-1]] = dict(R1=r1 * r1_scale, R2=1e15, C=1e-3)
    inlet_node = 0
    inlet_v = nodes.pop(inlet_node)
    assert len(inlet_v) == 1
    inlets = {inlet_v[0][0]: 1.0}
    nl = []
    for key, ends in nodes.items():
        if len(ends) == 1:
            if ends[0][0] in outlets:
                continue
            raise ValueError(f"dangling end {ends} at node {key}")
        nl.append(ends)
    # terminal nodes (numbers with a single right end that are outlets) were popped above via len==1 branch
    net = build_network(vessels, nl, inlets, outlets, rho=RHO, mu=mu, sigma=0.5, pv=pout, dx_target=dx,
                        wtab=inflow_table(), period=T)
    return net, chains, mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pout", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=2e-3)
    ap.add_argument("--sub", type=float, default=1.0)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--max-cycles", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net, chains, mesh = build(a.pout, a.dx, a.sub, mu=MU * a.mu_scale)
    print("sub-vessels", len(net.names), "cells", int(net.ncell.sum()), flush=True)
    net.init_uniform_pressure(12000.0 + a.pout)
    net.Pc[:] = a.pout  # pure resistor to p_out: capacitor state must start at the outlet pressure
    dt0 = net.stable_dt(0.8, u_max=3.0, a_ratio=1.2)
    spc = int(math.ceil(T / dt0 / 827)) * 827
    dt = T / spc
    prev, change = None, None
    for cyc in range(1, a.max_cycles + 1):
        net.advance(spc, dt)
        cur = net.A / np.repeat(net.A0, net.ncell)
        if prev is not None:
            change = float(np.linalg.norm(cur - prev) / np.linalg.norm(cur))
            print(f"cycle {cyc} change {change:.2e}", flush=True)
            if change < 1e-5:
                break
        prev = cur
    per = spc // 827
    P = np.zeros((827, len(chains)))
    Q = np.zeros((827, len(chains)))
    for i in range(827):
        net.advance(per, dt)
        for s, names in chains.items():
            mid = names[len(names) // 2]
            v = net.vessel_index(mid)
            j = int(net.off[v]) + int(net.ncell[v]) // 2
            P[i, s - 1] = net.bco[v] * (math.sqrt(net.A[j]) - math.sqrt(net.A0[v])) / net.A0[v] / 1e3
            Q[i, s - 1] = net.Q[j] * 1e6
    np.savez(a.out, P=P, Q=Q, cycles=cyc, change=change, vol_in=net.acc[0], vol_out=net.acc[1], dt=dt)
    print("saved", a.out, "periodic change", change)


if __name__ == "__main__":
    main()
