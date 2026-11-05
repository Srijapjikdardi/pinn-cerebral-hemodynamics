"""Mesh convergence of the verified CoW solver. Usage: python scripts/gate0/cow_mesh_study.py
Reference = finest dx. Reports per-vessel peak-pressure diff and waveform rel-L2 (P, Q) at mid-vessel."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.cow_network import build_cow
from physics.solver_1d import simulate_periodic

DX = [4e-3, 2e-3, 1e-3, 0.5e-3]
res = {}
for dx in DX:
    net, info = build_cow((), dx_target=dx)
    res[dx] = simulate_periodic(net, info["p_init"], n_out=80, tol=1e-6, max_cycles=40)
    d = res[dx]["diagnostics"]
    print(f"dx={dx*1e3}mm cycles={d['cycles_to_converge']} conv={d['converged']} wall={d['wall_s']:.1f}s", flush=True)
ref = res[DX[-1]]
names = ref["names"]
def at_mid(r, n, f):
    x = r[n]["x"]; L = x[-1] + (x[1]-x[0]) / 2
    xi = 0.5 * L
    return np.array([np.interp(xi, x, r[n][f][i]) for i in range(r[n][f].shape[0])])
qscale = max(np.linalg.norm(at_mid(ref, n, "Q")) for n in names)
print("\nworst over vessels vs finest (Q denominator floored at 1% of the largest vessel):")
print(f"{'dx_mm':>6s} {'Pmax diff %':>12s} {'P relL2 %':>10s} {'Q relL2 %':>10s}  worst vessel (Q)")
for dx in DX[:-1]:
    pm, pl, ql, wq = 0, 0, 0, ""
    for n in names:
        a, b = at_mid(res[dx], n, "P"), at_mid(ref, n, "P")
        pm = max(pm, abs(a.max()-b.max())/abs(b.max()))
        pl = max(pl, np.linalg.norm(a-b)/np.linalg.norm(b))
        a, b = at_mid(res[dx], n, "Q"), at_mid(ref, n, "Q")
        e = np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-2*qscale)  # symmetric ACoA has Q~0: floor the denominator
        if e > ql: ql, wq = e, n
    print(f"{dx*1e3:6.1f} {100*pm:12.3f} {100*pl:10.3f} {100*ql:10.3f}  {wq}")
