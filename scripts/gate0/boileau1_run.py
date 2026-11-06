"""Boileau 2015 case 1 (single pulse, 10 m tube) vs the six codes' pressure profiles (supplement Benchmark1_SinglePulseModel).
Usage: python scripts/gate0/boileau1_run.py /path/to/Benchmark1_SinglePulseModel
Compares pressure vs distance at t = 0.1..1.5 s (inviscid; viscous with mu = 4 mPa s and friction scale for zeta = 2 and zeta = 9)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.solver_1d import build_network  # noqa: E402

RHO, A0, H, E = 1050.0, math.pi * 1e-4, 1.5e-3, 4.0e5
BETA = math.sqrt(math.pi) * H * E / 0.75
C0 = math.sqrt(BETA / (2 * RHO * math.sqrt(A0)))
Z0 = RHO * C0 / A0
TIMES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]


def run(mu, dx=2e-3):
    n_tab, period = 30000, 3.0
    tt = np.arange(n_tab) / n_tab * period
    wtab = 1e-6 * np.exp(-10000.0 * (tt - 0.05) ** 2)
    v = dict(name="a", L=10.0, r0=math.sqrt(A0 / math.pi), h=H, E=E)
    net = build_network([v], [], {"a": 1.0}, {"a": dict(R1=Z0, R2=1e12, C=1e-6)}, rho=RHO, mu=mu, sigma=0.5, pv=0.0,
                        wtab=wtab, period=period, dx_target=dx)
    net.init_uniform_pressure(0.0)
    dt0 = net.stable_dt(0.8, u_max=0.01, a_ratio=1.001)
    nper = int(math.ceil(0.1 / dt0))
    dt = 0.1 / nper
    x = net.x_centres("a")
    out = []
    for i, _ in enumerate(TIMES):
        net.advance(nper if i == 0 else 2 * nper, dt)  # TIMES = 0.1, 0.3, ...: first leg 0.1 s, then 0.2 s
        out.append(net.pressure(net.A, "a") / 1e3)  # kPa
    return x, np.array(out).T


def main(root):
    for variant, scales in (("inviscid", [("mu=0", 0.0)]), ("viscous", [("zeta=2 (8 pi mu)", 1.0), ("zeta=9 (x2.53)", 2.53)])):
        ref = {}
        for s in ("DCG", "LCG", "FEM", "FVM", "McC", "STM"):
            d = np.loadtxt(f"{root}/NumData/{s}_{variant}.txt")
            ref[s] = d
        xr = ref["FVM"][:, 0]
        on = {s: np.array([np.interp(xr, ref[s][:, 0], ref[s][:, k + 1]) for k in range(len(TIMES))]).T for s in ref}
        mean = np.mean(list(on.values()), 0)
        spread = np.mean([np.linalg.norm(on[s] - mean) / np.linalg.norm(mean) for s in on])
        print(f"\n{variant}: six-code relL2 spread about their mean {spread*100:.2f}% (peak at t=0.1/1.5 s: {mean[:,0].max():.4f}/{mean[:,-1].max():.4f} kPa)")
        for label, sc in scales:
            x, P = run(4e-3 * sc)
            Pi = np.array([np.interp(xr, x, P[:, k]) for k in range(len(TIMES))]).T
            err = np.linalg.norm(Pi - mean) / np.linalg.norm(mean)
            pk = [Pi[:, k].max() for k in (0, 3, 7)]
            print(f"  {label:18s} relL2 vs six-code mean {err*100:6.2f}%   peaks t=0.1/0.7/1.5: {pk[0]:.4f}/{pk[1]:.4f}/{pk[2]:.4f} kPa"
                  f"   (ref {mean[:,0].max():.4f}/{mean[:,3].max():.4f}/{mean[:,7].max():.4f})")


if __name__ == "__main__":
    main(sys.argv[1])
