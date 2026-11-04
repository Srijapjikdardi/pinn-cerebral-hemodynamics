"""Gate 0: volume budget of a dataset file.

Usage (from repo root):  python scripts/gate0/dataset_budget.py data/synthetic_demo.h5 [patient_index]

Prints, over all patients: inflow volume, MCA outflow, change in stored volume, and the
stored/inflow fraction. With a patient index also prints the per-segment budget
(in = int Q[0] dt, out = int Q[-1] dt, dV = change of int A dx).
A mass-conserving solver has in - out - dV ~ 0 for every segment.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.network_topology import SEGMENTS  # noqa: E402

INLETS = ("ICA_L", "ICA_R", "PCA_P1_L", "PCA_P1_R")


def load(g):
    return {sk[len("segment_"):]: {f: g[sk][f][:] for f in g[sk].keys()}
            for sk in g.keys() if sk.startswith("segment_")}


def main(path, index=None):
    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys())
        fr, vin, vdv, vout = [], [], [], []
        for k in keys:
            s = load(h5[k])
            t = s["ICA_L"]["t"]
            q_in = np.trapezoid(sum(s[n]["Q"][:, 0] for n in INLETS if n in s), t)
            q_out = np.trapezoid(sum(s[n]["Q"][:, -1] for n in ("MCA_L", "MCA_R") if n in s), t)
            stored = sum(np.trapezoid(sd["A"], dx=sd["x"][1] - sd["x"][0], axis=1) for sd in s.values())
            dv = stored[-1] - stored[0]
            vin.append(q_in * 1e6); vout.append(q_out * 1e6); vdv.append(dv * 1e6); fr.append(dv / q_in)
        print(f"{path}: {len(keys)} patients")
        print(f"inflow   [mL] mean {np.mean(vin):.3f}")
        print(f"MCA out  [mL] mean {np.mean(vout):.3f}")
        print(f"stored   [mL] mean {np.mean(vdv):.4f}")
        print(f"stored/inflow: mean {np.mean(fr):.4f}  min {np.min(fr):.4f}  max {np.max(fr):.4f}")
        if index is not None:
            s = load(h5[keys[index]])
            print(f"\n{keys[index]} per-segment budget [mL]")
            print(f"{'seg':9s} {'in':>9s} {'out':>9s} {'dV':>9s} {'in-out-dV':>10s}")
            for name, *_ in SEGMENTS:
                if name not in s:
                    continue
                sd = s[name]
                t = sd["t"]
                qi = np.trapezoid(sd["Q"][:, 0], t) * 1e6
                qo = np.trapezoid(sd["Q"][:, -1], t) * 1e6
                vol = np.trapezoid(sd["A"], dx=sd["x"][1] - sd["x"][0], axis=1)
                dv = (vol[-1] - vol[0]) * 1e6
                print(f"{name:9s} {qi:9.4f} {qo:9.4f} {dv:9.4f} {qi - qo - dv:10.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
