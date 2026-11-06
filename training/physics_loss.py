"""Differentiable physics penalties on the surrogate's normalised output (N,V,2,NT,NX), channels (A/A0-1, Q).
Terms: (a) continuity A_t + Q_x = 0 by finite differences (periodic central difference in time, central in x);
(b) junction mass sum(sign * Q at vessel ends) = 0. Both normalised by flow scales so they are O(1) when violated at the
size of the signal. These are the quantities reliability signals S1 and part of S3 are built from, so a surrogate trained
with them tests whether the signals survive physics-informed training."""
import math

import numpy as np
import torch

from data.cow_dataset import GEO, VESSELS
from reliability.signals import DX_SOLVER, NODES


def build(meta, nt, nx):
    V = len(VESSELS)
    N = len(meta)
    a0 = np.zeros((N, V), np.float32)
    for n, m in enumerate(meta):
        for i, v in enumerate(VESSELS):
            if v in m["vessels"]:
                a0[n, i] = math.pi * (GEO["vessels"][v]["r_cm"] * 1e-2 * m["r_scale"][v]) ** 2
    L = np.array([GEO["vessels"][v]["length_cm"] * 1e-2 for v in VESSELS], np.float32)
    ncell = np.maximum(3, np.round(L / DX_SOLVER))
    h = (L - L / ncell) / (nx - 1)
    dt = np.array([m["period"] / nt for m in meta], np.float32)
    nodes = []
    for node in NODES.values():
        w = np.zeros((V, 2), np.float32)  # sign per vessel end (x=0 end index 0, x=L end index 1)
        for v, e in node:
            w[VESSELS.index(v), e] = 1.0 if e == 1 else -1.0
        nodes.append(w)
    return dict(a0=torch.as_tensor(a0), h=torch.as_tensor(h), L=torch.as_tensor(L), dt=torch.as_tensor(dt),
                nodes=torch.as_tensor(np.stack(nodes)))


def loss(out_n, ys, idx, M, P):
    """out_n: normalised prediction (B,V,2,NT,NX); ys: (1,V,2,1,1) scale; idx: sample indices; M: (N,V) present mask."""
    pred = out_n * ys
    A, Q = pred[:, :, 0], pred[:, :, 1]  # A here is A/A0 - 1 (B,V,NT,NX); Q physical (m3/s)
    m = M[idx]  # (B,V)
    a0 = P["a0"][idx]  # (B,V)
    dt = P["dt"][idx]  # (B,)
    At = (torch.roll(A, -1, 2) - torch.roll(A, 1, 2)) / (2 * dt[:, None, None, None]) * a0[:, :, None, None]
    h = P["h"][None, :, None, None]
    Qx = (Q[..., 2:] - Q[..., :-2]) / (2 * h)
    res = At[..., 1:-1] + Qx
    denom = (ys[:, :, 1, :, :] .squeeze(-1).squeeze(-1) / P["L"][None, :])[:, :, None, None].clamp_min(1e-12)
    mass = ((res / denom) ** 2 * m[:, :, None, None]).sum() / (m.sum() * res.shape[2] * res.shape[3]).clamp_min(1.0)
    ends = torch.stack([Q[..., 0], Q[..., -1]], -1)  # (B,V,NT,2)
    nodes = P["nodes"]  # (K,V,2)
    resj = torch.einsum("bvte,kve->bkt", ends * m[:, :, None, None], nodes)
    cnt = (m[:, None, :, None] * nodes.abs()[None]).sum((2, 3)) >= 2  # node has >= 2 present ends
    qs = ys[:, :, 1].max()
    junc = ((resj / qs) ** 2 * cnt[:, :, None]).sum() / (cnt.sum() * resj.shape[2]).clamp_min(1.0)
    return mass, junc


def project_junctions(out_n, ys, idx, M, P):
    """Exact junction-mass projection (hard constraint). For each junction with >= 2 present ends, subtract the signed flow
    residual r(t) = sum_e sign_e Q_e equally from the ends (end correction -sign_e r/n), spread along each vessel with a linear
    ramp (full at the junction end, zero at the far end). After projection sum_e sign_e Q_e(end grid point) = 0 exactly.
    out_n: normalised prediction (B,V,2,NT,NX); returns normalised prediction of the same shape."""
    pred = out_n * ys
    Q = pred[:, :, 1]  # (B,V,NT,NX)
    m = M[idx]
    nodes = P["nodes"]  # (K,V,2)
    nx = Q.shape[-1]
    ends = torch.stack([Q[..., 0], Q[..., -1]], -1)  # (B,V,NT,2)
    pres = m[:, None, :, None] * nodes.abs()[None]  # (B,K,V,2) present ends of node k
    n_present = pres.sum((2, 3))  # (B,K)
    resj = torch.einsum("bvte,kve->bkt", ends * m[:, :, None, None], nodes)
    share = torch.where(n_present >= 2, 1.0 / n_present.clamp_min(1.0), torch.zeros_like(n_present))  # (B,K)
    # correction per vessel end: -sign * r * share, summed over nodes (an end belongs to at most one node)
    corr = -torch.einsum("bkt,kve,bk->bvte", resj, nodes, share)  # (B,V,NT,2)
    ramp0 = torch.linspace(1.0, 0.0, nx, device=Q.device)
    ramp1 = torch.linspace(0.0, 1.0, nx, device=Q.device)
    dq = corr[..., 0:1] * ramp0 + corr[..., 1:2] * ramp1  # (B,V,NT,NX)
    pred = pred.clone()
    pred[:, :, 1] = Q + dq * m[:, :, None, None]
    return pred / ys


def junction_residual_per_sample(pred, idx, M, P):
    """Per-sample junction-mass violation of a physical-unit prediction (B,V,2,NT,NX), normalised like signals.S1:
    mean over junctions with >= 2 present ends of mean_t |sum_e sign_e Q_e| / max |Q| over present vessels. Used on the raw
    network output before project_junctions, i.e. the size of the correction the hard-constraint layer had to apply."""
    Q = pred[:, :, 1]
    m = M[idx]
    ends = torch.stack([Q[..., 0], Q[..., -1]], -1) * m[:, :, None, None]
    nodes = P["nodes"]
    resj = torch.einsum("bvte,kve->bkt", ends, nodes).abs().mean(-1)  # (B,K)
    cnt = ((m[:, None, :, None] * nodes.abs()[None]).sum((2, 3)) >= 2).float()
    qscale = (Q.abs() * m[:, :, None, None]).amax(dim=(1, 2, 3)).clamp_min(1e-30)
    return (resj * cnt).sum(1) / cnt.sum(1).clamp_min(1.0) / qscale
