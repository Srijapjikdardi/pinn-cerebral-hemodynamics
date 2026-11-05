"""Time the verified CoW solver. Usage: python scripts/gate0/time_cow.py [dx_mm] [repeats]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.cow_network import build_cow
from physics.solver_1d import simulate_periodic

dx = float(sys.argv[1]) * 1e-3 if len(sys.argv) > 1 else 1e-3
rep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
net, info = build_cow((), dx_target=dx)
simulate_periodic(net, info["p_init"], n_out=80, tol=1e-5)  # warm-up/JIT
ts = []
for _ in range(rep):
    net, info = build_cow((), dx_target=dx)
    o = simulate_periodic(net, info["p_init"], n_out=80, tol=1e-5)
    ts.append(o["diagnostics"]["wall_s"])
d = o["diagnostics"]
print(f"dx={dx*1e3:.1f}mm cells={int(net.ncell.sum())} cycles={d['cycles_to_converge']} "
      f"wall mean {sum(ts)/len(ts):.2f}s min {min(ts):.2f} max {max(ts):.2f} cfl {d['max_courant']:.2f}")
