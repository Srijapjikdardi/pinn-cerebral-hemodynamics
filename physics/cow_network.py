"""Circle of Willis network builder. Variants are real graph changes: absent vessels are removed, junctions
re-formed, and vessels no longer reachable from an inlet are pruned. Geometry and provenance live in
configs/geometry_alastruey2007.yaml."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from physics.solver_1d import build_network

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "geometry_alastruey2007.yaml"
MMHG = 133.322
PULSATILITY = 0.5


def load_geometry(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def inflow_waveform(n=400):
    t = np.arange(n) / n * 0.8
    w = 0.35 + np.exp(-0.5 * ((t - 0.15) / 0.07) ** 2)
    w = w / w.mean()
    return 1.0 + PULSATILITY * (w - 1.0)  # peak/mean 1.69 at 0.5 (A: cerebral inflow is damped vs. the raw pulse)


def reachable_vessels(geo, absent):
    present = {v for v in geo["vessels"] if v not in set(absent)}
    nodes = [[(v, e) for v, e in n if v in present] for n in geo["topology"]["nodes"].values()]
    nodes = [n for n in nodes if len(n) >= 1]
    reach = {v for v in geo["topology"]["inlets"] if v in present}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if any(v in reach for v, _ in n):
                for v, _ in n:
                    if v not in reach:
                        reach.add(v)
                        changed = True
    return reach, nodes


def build_cow(absent=(), geo=None, dx_target=1.0e-3, hypo=None, E_scale=None, inflow_scale=None,
              R_scale=1.0, period=None):
    """absent: vessel names to remove. hypo: {name: radius_scale} (h scales with r).
    E_scale: {name: factor}. inflow_scale: {inlet name: factor}. R_scale: terminal R_total factor.
    period: cardiac period in s (default from config). Returns (net, info)."""
    geo = geo or load_geometry()
    absent = tuple(absent)
    for a in absent:
        if a not in geo["variable_vessels"]:
            raise ValueError(f"{a} is not a variable vessel")
    keep, nodes = reachable_vessels(geo, absent)
    names = [v for v in geo["vessels"] if v in keep]
    c = geo["constants"]
    vessels = []
    for n in names:
        g = geo["vessels"][n]
        s = (hypo or {}).get(n, 1.0)
        vessels.append(dict(name=n, L=g["length_cm"] * 1e-2, r0=g["r_cm"] * 1e-2 * s,
                            h=g["h_cm"] * 1e-2 * s, E=g["E"] * 1e6 * (E_scale or {}).get(n, 1.0)))
    nodes2 = []
    for nd in nodes:
        nd = [(v, e) for v, e in nd if v in keep]
        if len(nd) >= 2:
            nodes2.append(nd)
        elif len(nd) == 1:
            raise ValueError(f"dead end: vessel end {nd[0]} would be unterminated")
    inlets_cfg = [v for v in geo["topology"]["inlets"] if v in keep]
    mean = {"ICA": geo["inflow"]["ICA_mean_mL_s"], "BA": geo["inflow"]["BA_mean_mL_s"]}
    inlets = {v: (mean["BA"] if v == "BA" else mean["ICA"]) * 1e-6 * (inflow_scale or {}).get(v, 1.0)
              for v in inlets_cfg}
    outlets = {}
    for v in geo["topology"]["outlets"]:
        if v in keep:
            g = geo["vessels"][v]
            outlets[v] = dict(R_total=g["R_total_1e9"] * 1e9 * R_scale, C=g["C_1e-10"] * 1e-10)
    # adjust R_total of hypoplastic outlets is not modelled; Z0 check happens in build_network
    net = build_network(vessels, nodes2, inlets, outlets, rho=c["rho_kg_m3"], mu=c["mu_mPa_s"] * 1e-3,
                        sigma=c["sigma"], pv=c["venous_pressure_mmHg"] * MMHG, dx_target=dx_target,
                        wtab=inflow_waveform(), period=period or c["period_s"])
    info = dict(absent=absent, vessels=names, p_init=c["p_init_mmHg"] * MMHG, period=period or c["period_s"],
                n_pruned=len(geo["vessels"]) - len(names) - len(absent))
    return net, info
