"""Generate CoW dataset with the verified solver (physics/solver_1d.py).

Each sample = (topology variant, randomised physiology) -> periodic-state fields of every present vessel on a fixed
grid (NX points, NT times) in float32. Variant = set of absent variable vessels (96 valid of 128; see variants()).
Usage (repo root):  python data/generate_cow.py --n 20 --out data/cow_pilot.h5 --seed 0 [--workers 2]
Sampling is deterministic given (seed, index). Variants are drawn uniformly over the valid list.
"""
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics.cow_network import build_cow, load_geometry  # noqa: E402
from physics.solver_1d import simulate_periodic  # noqa: E402

NX, NT, DX = 21, 80, 2.0e-3
RAD_SD, E_SD, INFLOW_SD, R_SD = 0.08, 0.12, 0.12, 0.12  # lognormal sigmas (assumed ranges, A)
PERIOD_RANGE = (0.6, 1.0)
# Mean outflow demand per terminal, mL/s (Alastruey 2007 Table 2, normal case). Inflow is scaled by the fraction of
# demand whose terminals remain connected (A: variants that prune territory do not push full flow into fewer outlets).
DEMAND = {"MCA": 1.73, "A2": 1.16, "P2": 0.90}


def demand_fraction(vessels):
    full = 2 * sum(DEMAND.values())
    kept = sum(DEMAND[v.split("_")[0]] for v in vessels if v.split("_")[0] in DEMAND)
    return kept / full


def variants():
    geo = load_geometry()
    vv = geo["variable_vessels"]
    ok = []
    for k in range(len(vv) + 1):
        for c in itertools.combinations(vv, k):
            try:
                build_cow(c, geo=geo, dx_target=5e-3)
                ok.append(c)
            except ValueError:
                pass
    return ok


def sample_params(seed, idx, variant_list):
    rng = np.random.default_rng([seed, idx])
    geo = load_geometry()
    absent = variant_list[rng.integers(len(variant_list))]
    ln = lambda sd: float(np.exp(rng.normal(0.0, sd)))  # noqa: E731
    return dict(
        absent=list(absent),
        r_scale={v: ln(RAD_SD) for v in geo["vessels"]},
        E_scale={v: ln(E_SD) for v in geo["vessels"]},
        inflow_scale={v: ln(INFLOW_SD) for v in geo["topology"]["inlets"]},
        R_scale=ln(R_SD),
        period=float(rng.uniform(*PERIOD_RANGE)),
    )


def run_sample(args):
    seed, idx, variant_list = args
    prm = sample_params(seed, idx, variant_list)
    _, pre = build_cow(prm["absent"], dx_target=5e-3)
    frac = demand_fraction(pre["vessels"])
    inflow = {k: v * frac for k, v in prm["inflow_scale"].items()}
    net, info = build_cow(prm["absent"], dx_target=DX, hypo=prm["r_scale"], E_scale=prm["E_scale"],
                          inflow_scale=inflow, R_scale=prm["R_scale"], period=prm["period"])
    out = simulate_periodic(net, info["p_init"], n_out=NT, tol=1e-5, max_cycles=40)
    d = out["diagnostics"]
    fields = {}
    for n in info["vessels"]:
        x = out[n]["x"]
        L = x[-1] + (x[1] - x[0]) / 2
        xs = np.linspace(0.5 * (x[1] - x[0]), L - 0.5 * (x[1] - x[0]), NX)
        fields[n] = {f: np.stack([np.interp(xs, x, out[n][f][i]) for i in range(NT)]).astype(np.float32)
                     for f in ("A", "Q", "P")}
    pall = np.concatenate([fd["P"].ravel() for fd in fields.values()]) / 133.322
    meta = dict(idx=idx, p_sys_mmHg=float(pall.max()), p_dia_mmHg=float(pall.min()),
                physiological=bool(pall.max() <= 180.0 and pall.min() >= 40.0),  # A: filter flag, not a gate
                 absent=prm["absent"], vessels=info["vessels"], n_pruned=info["n_pruned"], demand_fraction=frac,
                converged=bool(d["converged"]), cycles=int(d["cycles_to_converge"]),
                cycle_change=float(d["cycle_change"]), max_courant=float(d["max_courant"]),
                wall_s=float(d["wall_s"]), period=prm["period"], R_scale=prm["R_scale"],
                r_scale=prm["r_scale"], E_scale=prm["E_scale"], inflow_scale=prm["inflow_scale"])
    return idx, meta, fields, out["t"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/cow_pilot.h5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    vl = variants()
    t0 = time.time()
    with h5py.File(a.out, "w") as h5, ProcessPoolExecutor(a.workers) as ex:
        h5.attrs["seed"] = a.seed
        h5.attrs["nx"], h5.attrs["nt"], h5.attrs["dx_solver"] = NX, NT, DX
        h5.attrs["variants"] = json.dumps(vl)
        for k, (idx, meta, fields, t) in enumerate(ex.map(run_sample, [(a.seed, i, vl) for i in range(a.n)])):
            g = h5.create_group(f"s{idx:05d}")
            g.attrs["meta"] = json.dumps(meta)
            g.create_dataset("t", data=t)
            for n, fd in fields.items():
                for f, arr in fd.items():
                    g.create_dataset(f"{n}/{f}", data=arr, compression="gzip", compression_opts=4)
            if (k + 1) % 10 == 0:
                print(f"{k+1}/{a.n} {time.time()-t0:.0f}s", flush=True)
    print(f"done {a.n} samples in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
