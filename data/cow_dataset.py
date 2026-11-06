"""Load data/cow_*.h5 into dense arrays.

X  (N, 42): absent mask(7) | r_scale(15) | E_scale(15) | inflow_scale(3: ICA_L, ICA_R, BA) | R_scale | period
Y  (N, 15, 2, NT, NX): channels (A/A0 - 1, Q); absent vessels zero
M  (N, 15): 1 if vessel present
Vessel order = configs/geometry_alastruey2007.yaml. Pruned vessels count as absent (M=0).
"""
import json

import h5py
import numpy as np

from physics.cow_network import load_geometry

GEO = load_geometry()
VESSELS = list(GEO["vessels"])
VAR = GEO["variable_vessels"]
INLETS = GEO["topology"]["inlets"]


def a0_of(name, r_scale=1.0):
    return np.pi * (GEO["vessels"][name]["r_cm"] * 1e-2 * r_scale) ** 2


def load(path, physiological_only=False, indices=None):
    """indices: optional positions in sorted key order to load (memory-light partial load)."""
    X, Y, M, meta = [], [], [], []
    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys())
        for k in (keys if indices is None else [keys[i] for i in indices]):
            g = h5[k]
            m = json.loads(g.attrs["meta"])
            if physiological_only and not m["physiological"]:
                continue
            pres = np.array([v in m["vessels"] for v in VESSELS], float)
            x = np.concatenate([
                [float(v in m["absent"]) for v in VAR],
                [m["r_scale"][v] for v in VESSELS],
                [m["E_scale"][v] for v in VESSELS],
                [m["inflow_scale"][v] for v in INLETS],
                [m["R_scale"], m["period"]],
            ])
            nt, nx = g["t"].shape[0], g[f"{m['vessels'][0]}/A"].shape[1]
            y = np.zeros((len(VESSELS), 2, nt, nx), np.float32)
            for i, v in enumerate(VESSELS):
                if pres[i]:
                    a0 = a0_of(v, m["r_scale"][v])
                    y[i, 0] = g[f"{v}/A"][:] / a0 - 1.0
                    y[i, 1] = g[f"{v}/Q"][:]
            X.append(x); Y.append(y); M.append(pres); meta.append(m)
    return np.array(X, np.float32), np.stack(Y), np.array(M, np.float32), meta
