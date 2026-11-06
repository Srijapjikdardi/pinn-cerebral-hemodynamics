"""Effect of the friction closure on CoW fields. Re-solves N dataset samples (same parameters as data/cow_v1.h5, same seed)
with friction_scale 1 (parabolic, dataset) and 2.53 (zeta = 9, Boileau-style); reports relative L2 differences.
Usage: python scripts/gate0/friction_sensitivity.py N [out.npz]"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.generate_cow import DX, NT, NX, demand_fraction, sample_params, variants  # noqa: E402
from physics.cow_network import build_cow  # noqa: E402
from physics.solver_1d import simulate_periodic  # noqa: E402

VL = variants()


def solve(args):
    idx, scale = args
    prm = sample_params(1, idx, VL)  # seed 1 = cow_v1.h5
    _, pre = build_cow(prm["absent"], dx_target=5e-3)
    frac = demand_fraction(pre["vessels"])
    net, info = build_cow(prm["absent"], dx_target=DX, hypo=prm["r_scale"], E_scale=prm["E_scale"],
                          inflow_scale={k: v * frac for k, v in prm["inflow_scale"].items()},
                          R_scale=prm["R_scale"], period=prm["period"])
    net.friction_scale = scale
    out = simulate_periodic(net, info["p_init"], n_out=NT, tol=1e-5, max_cycles=40)
    return {n: (out[n]["A"], out[n]["Q"], out[n]["P"]) for n in info["vessels"]}, out["diagnostics"]["converged"]


def main():
    n = int(sys.argv[1])
    ids = list(range(n))
    with ProcessPoolExecutor(2) as ex:
        base = list(ex.map(solve, [(i, 1.0) for i in ids]))
        z9 = list(ex.map(solve, [(i, 2.53) for i in ids]))
    rows = []
    for (a, ca), (b, cb) in zip(base, z9):
        r = []
        for k in range(3):  # A, Q, P
            num = sum(np.sum((a[v][k] - b[v][k]) ** 2) for v in a)
            den = sum(np.sum(a[v][k] ** 2) for v in a)
            r.append(np.sqrt(num / den))
        pa = np.concatenate([a[v][2].ravel() for v in a]); pb = np.concatenate([b[v][2].ravel() for v in b])
        r += [abs(pb.max() - pa.max()) / pa.max(), abs(pb.min() - pa.min()) / pa.min()]
        rows.append(r)
    R = np.array(rows) * 100
    print(f"N={n}, all converged: {all(c for _, c in base) and all(c for _, c in z9)}")
    for k, name in enumerate(["A relL2", "Q relL2", "P relL2", "P max diff", "P min diff"]):
        print(f"{name:11s} % : median {np.median(R[:, k]):.3f}  p90 {np.percentile(R[:, k], 90):.3f}  max {R[:, k].max():.3f}")
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], R=R)


if __name__ == "__main__":
    main()
