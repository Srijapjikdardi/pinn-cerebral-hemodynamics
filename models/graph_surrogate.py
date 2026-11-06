"""Vessel-graph surrogate: second architecture family for the reliability study.

Each vessel is a graph node. Node features: present flag, standardised per-vessel inputs (r_scale, E_scale), static
geometry (log length, radius, wall thickness, E), role (inlet / outlet / internal), inlet scale for inlets, and the global
inputs (R_scale, period). Edges connect vessels that share a junction (from configs/geometry_alastruey2007.yaml); edges to
or from absent vessels are masked, so a new topology is a new graph, not an unseen input pattern. K rounds of masked
message passing, then a decoder shared across vessels maps each node embedding (plus a global mean-pooled readout and a learned vessel
embedding) to its
(A/A0-1, Q) field of shape (2, NT, NX).

Inductive bias differs from the MLP baseline (which maps one flat input vector to all fields): weights are shared over
vessels and the topology enters through connectivity. The model reads the same inputs, so the reliability comparison is fair.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from data.cow_dataset import GEO, INLETS, VESSELS

V = len(VESSELS)


def _static():
    outlets = set(GEO["topology"]["outlets"])
    f = []
    for v in VESSELS:
        g = GEO["vessels"][v]
        f.append([math.log(g["length_cm"]), g["r_cm"] * 10, g["h_cm"] * 10, g["E"],
                  float(v in INLETS), float(v in outlets), float(v not in INLETS and v not in outlets)])
    f = np.array(f, np.float32)
    f[:, :4] = (f[:, :4] - f[:, :4].mean(0)) / (f[:, :4].std(0) + 1e-6)
    adj = np.zeros((V, V), np.float32)
    for node in GEO["topology"]["nodes"].values():
        idx = [VESSELS.index(v) for v, _ in node]
        for i in idx:
            for j in idx:
                if i != j:
                    adj[i, j] = 1.0
    return torch.as_tensor(f), torch.as_tensor(adj)


class GraphSurrogate(nn.Module):
    """forward(xn, m): xn standardised inputs (B, 42) in cow_dataset.load layout; m present mask (B, V).
    Returns flat output (B, V*2*NT*NX) so it is a drop-in replacement for the MLP in training.train_surrogate.fit."""

    def __init__(self, nt, nx, width=128, rounds=4):
        super().__init__()
        stat, adj = _static()
        self.register_buffer("stat", stat)
        self.register_buffer("adj", adj)
        self.nt, self.nx = nt, nx
        nvar = len(GEO["variable_vessels"])
        self.i_r = slice(nvar, nvar + V)
        self.i_E = slice(nvar + V, nvar + 2 * V)
        self.i_in = slice(nvar + 2 * V, nvar + 2 * V + len(INLETS))
        self.i_glob = slice(nvar + 2 * V + len(INLETS), nvar + 2 * V + len(INLETS) + 2)
        self.inlet_pos = [VESSELS.index(v) for v in INLETS]
        din = 1 + 2 + stat.shape[1] + 1 + 2
        self.vemb = nn.Parameter(torch.randn(V, 16) * 0.1)
        self.enc = nn.Sequential(nn.Linear(din + 16, width), nn.GELU(), nn.Linear(width, width))
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
                                  for _ in range(rounds)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(rounds)])
        self.dec = nn.Sequential(nn.Linear(2 * width + 16, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, 2 * nt * nx))

    def forward(self, xn, m):
        B = xn.shape[0]
        inl = torch.zeros(B, V, 1, device=xn.device)
        inl[:, self.inlet_pos, 0] = xn[:, self.i_in]
        feat = torch.cat([m[..., None], xn[:, self.i_r, None], xn[:, self.i_E, None],
                          self.stat[None].expand(B, -1, -1), inl,
                          xn[:, None, self.i_glob].expand(-1, V, -1), self.vemb[None].expand(B, -1, -1)], -1)
        h = self.enc(feat) * m[..., None]
        a = self.adj[None] * m[:, :, None] * m[:, None, :]  # (B,V,V) edges among present vessels
        deg = a.sum(-1, keepdim=True).clamp_min(1.0)
        for msg, norm in zip(self.msg, self.norm):
            agg = torch.bmm(a, h) / deg
            h = norm(h + msg(torch.cat([h, agg], -1))) * m[..., None]
        g = (h.sum(1) / m.sum(1, keepdim=True).clamp_min(1.0))[:, None].expand(-1, V, -1)  # global readout over present vessels
        out = self.dec(torch.cat([h, g, self.vemb[None].expand(B, -1, -1)], -1)) * m[..., None]
        return out.view(B, V, 2, self.nt, self.nx).flatten(1)
