"""Physics-based reliability signals for a predicted CoW sample. Inputs are in the dataset's normalised
layout: pred (V,2,NT,NX) with channels (A/A0-1, Q), plus the sample meta. All signals are scalars per sample,
larger = less reliable. Signals use predicted fields only (plus known inputs: geometry, inflow, period).

S1 junction mass      : mean_t |sum_k sign_k Q_k| / mean flow scale, at vessel-end grid points of each junction
S2 junction pressure  : mean_t spread of total pressure (P + rho u^2/2) across incident vessel ends / pressure scale
S3 pde residual       : RMS of mass (A_t + Q_x) and momentum residuals by finite differences / field scales
S4 inlet mismatch     : mean_t |Q_pred(x_first, t) - Q_inflow(t)| / mean inflow, over inlets
S5 ensemble spread    : in reliability_eval (needs members)
S6 input OOD          : in reliability_eval (kNN distance in standardised input space)
Grid-end values sit half a solver cell from the true vessel end, so signals have a nonzero floor on exact solver
output; reliability_eval measures it.
"""
import math

import numpy as np

from data.cow_dataset import GEO, VESSELS, a0_of
from physics.cow_network import inflow_waveform, inlet_mean_mL_s

_C = GEO["constants"]
RHO, MU, SIGMA = float(_C["rho_kg_m3"]), float(_C["mu_mPa_s"]) * 1e-3, float(_C["sigma"])
DX_SOLVER = float(GEO["constants"].get("dx_target_m", 2.0e-3))  # solver cell size used to build the dataset (config-driven)
KR = float(_C.get("friction_scale", 1.0)) * 8 * math.pi * MU / RHO  # momentum friction coefficient of the dataset's solver
NODES = GEO["topology"]["nodes"]


def _vessel_consts(name, m):
    g = GEO["vessels"][name]
    s = m["r_scale"][name]
    r0, h = g["r_cm"] * 1e-2 * s, g["h_cm"] * 1e-2 * s
    E = g["E"] * 1e6 * m["E_scale"][name]
    a0 = math.pi * r0 ** 2
    b = math.sqrt(math.pi) * h * E / (1 - SIGMA ** 2)
    L = g["length_cm"] * 1e-2
    ncell = max(3, int(round(L / DX_SOLVER)))
    return dict(a0=a0, b=b, L=L, dxs=L / ncell, ncell=ncell)


def _fields(pred, name, m, nx):
    i = VESSELS.index(name)
    c = _vessel_consts(name, m)
    A = (pred[i, 0] + 1.0) * c["a0"]
    Q = pred[i, 1]
    P = c["b"] * (np.sqrt(A) - math.sqrt(c["a0"])) / c["a0"]
    h = (c["L"] - c["dxs"]) / (nx - 1)
    return A, Q, P, c, h


def signals(pred, m):
    """pred: (V,2,NT,NX) numpy. Returns dict of S1..S4 scalars."""
    nt, nx = pred.shape[2], pred.shape[3]
    pres = set(m["vessels"])
    F = {n: _fields(pred, n, m, nx) for n in VESSELS if n in pres}
    qscale = max(np.abs(f[1]).max() for f in F.values()) + 1e-30
    pscale = max(np.abs(f[2]).max() for f in F.values()) + 1e-30
    # S1, S2
    s1, s2 = [], []
    for node in NODES.values():
        ends = [(v, e) for v, e in node if v in pres]
        if len(ends) < 2:
            continue
        qs, pt = [], []
        for v, e in ends:
            A, Q, P, c, h = F[v]
            j = nx - 1 if e == 1 else 0
            sg = 1.0 if e == 1 else -1.0
            qs.append(sg * Q[:, j])
            u = Q[:, j] / A[:, j]
            pt.append(P[:, j] + 0.5 * RHO * u ** 2)
        s1.append(np.abs(np.sum(qs, 0)).mean() / qscale)
        pt = np.array(pt)
        s2.append((pt.max(0) - pt.min(0)).mean() / pscale)
    # S3
    period = m["period"]
    dt = period / nt
    rm, rq = [], []
    for v, (A, Q, P, c, h) in F.items():
        At = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * dt)  # periodic in time
        Qt = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0)) / (2 * dt)
        Qx = np.gradient(Q, h, axis=1)
        k3 = c["b"] / (3 * RHO * c["a0"])
        flux = Q ** 2 / A + k3 * A ** 1.5
        fx = np.gradient(flux, h, axis=1)
        mass = At + Qx
        mom = Qt + fx + KR * Q / A
        sA = np.abs(At).mean() + np.abs(Qx).mean() + 1e-30
        sQ = np.abs(Qt).mean() + np.abs(fx).mean() + np.abs(KR * Q / A).mean() + 1e-30
        sl = slice(1, nx - 1)
        rm.append(np.sqrt((mass[:, sl] ** 2).mean()) / sA)
        rq.append(np.sqrt((mom[:, sl] ** 2).mean()) / sQ)
    s3 = 0.5 * (np.mean(rm) + np.mean(rq))
    # S4
    wt = inflow_waveform(400, GEO)
    wn = len(wt)
    tt = (np.arange(nt) + 1) * dt  # solver recorded times; waveform phase unknown, compare phase-free via shape
    errs = []
    for v in GEO["topology"]["inlets"]:
        if v in F:
            Q0 = F[v][1][:, 0]
            target_mean = inlet_mean_mL_s(GEO, v) * 1e-6
            target_mean *= m["inflow_scale"][v] * m.get("demand_fraction", 1.0)
            # the solver evaluates the waveform at absolute time; compare after removing the cyclic shift
            w = target_mean * np.interp(np.linspace(0, 1, nt, endpoint=False), np.arange(wn) / wn, wt, period=1.0)
            best = min(np.abs(np.roll(w, k) - Q0).mean() for k in range(nt))
            errs.append(best / target_mean)
    return dict(S1_junction_mass=float(np.mean(s1)), S2_junction_pressure=float(np.mean(s2)),
                S3_pde_residual=float(s3), S4_inlet_mismatch=float(np.mean(errs)))
