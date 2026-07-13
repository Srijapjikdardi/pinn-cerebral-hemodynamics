"""
Phase 2 — Parametric Graph-PINN architecture.

Inputs per collocation point:
    x   : spatial position along the segment, normalized to [0, 1]
    t   : time within the cardiac cycle, normalized to [0, 1]
    seg_id : one-hot / embedding index for which of the 11 segments this point is on
    cond   : patient conditioning vector (diameters + presence mask + stiffness),
             see data/network_topology.py:PatientTopology.conditioning_vector()

Outputs: (A, Q, P) at that (x, t) for that patient.

Architecture: small MLP graph-encoder for `cond` -> embedding, concatenated with
[x, t, seg_id_embedding] and fed through a SIREN-style (sinusoidal activation)
trunk network, which tends to fit high-frequency pulsatile PDE solutions better
than ReLU/tanh MLPs (Sitzmann et al., 2020).

SEAM FOR UPGRADING TO A REAL GNN: if you outgrow the fixed 11-segment topology
(e.g. modeling MCAs/ACAs distal branches too), replace `GraphConditionEncoder`
with a `torch_geometric` message-passing encoder that takes the adjacency matrix
directly instead of a flattened vector. Everything downstream (the SIREN trunk)
is unaffected.
"""

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """SIREN layer: y = sin(omega_0 * (W x + b)), with the Sitzmann et al. init scheme."""

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class GraphConditionEncoder(nn.Module):
    """Encodes the flattened patient conditioning vector (diameters + presence mask +
    stiffness) into a fixed-size embedding. See module docstring for the GNN upgrade seam."""

    def __init__(self, cond_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class GPINN(nn.Module):
    def __init__(self, n_segments: int, cond_dim: int, model_cfg: dict):
        super().__init__()
        self.n_segments = n_segments
        self.seg_embed_dim = 8
        self.seg_embedding = nn.Embedding(n_segments, self.seg_embed_dim)

        self.graph_encoder = GraphConditionEncoder(
            cond_dim=cond_dim,
            hidden_dim=model_cfg["graph_encoder_hidden_dim"],
            out_dim=model_cfg["graph_encoder_out_dim"],
        )

        trunk_in_dim = 2 + self.seg_embed_dim + model_cfg["graph_encoder_out_dim"]  # x, t, seg, cond
        hidden = model_cfg["hidden_dim"]
        n_layers = model_cfg["n_hidden_layers"]
        omega_0 = model_cfg["siren_omega_0"]

        layers = [SineLayer(trunk_in_dim, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden, hidden, omega_0=omega_0))
        self.trunk = nn.Sequential(*layers)

        # A, Q get learned heads; P is computed analytically from A (see forward()
        # docstring) so it has no head of its own — this is deliberate, not an omission.
        self.head_A = nn.Linear(hidden, 1)
        self.head_Q = nn.Linear(hidden, 1)

        # --- Output scaling (critical, not cosmetic) ---
        # A raw linear+SIREN network naturally produces O(1) outputs. A single GLOBAL
        # physical scale for A/Q doesn't work here: CoW vessels span roughly a 10x range
        # in cross-sectional area (ACoA/PCoA ~1.5e-6 m^2 vs ICA ~1.6e-5 m^2). Using one
        # fixed A_scale meant early-training A predictions for small vessels were ~10x
        # too large relative to *their own* A0, which the tube law (sqrt(A/A0)) turns
        # into a huge residual for exactly those points — this was empirically confirmed
        # to dominate and destabilize the total loss. Fix: derive each point's own
        # characteristic area (A0_local) and flow (Q0_local) from its conditioning vector
        # (which already encodes per-segment diameter) and scale relative to that, so the
        # network only has to learn an O(1) *ratio*, not an absolute magnitude spanning
        # 10x across segments. Measured directly against the Phase 1 synthetic_demo.h5
        # dataset: real A/A0 only varies ~-4%/+10% over a cardiac cycle (arteries are
        # fairly stiff, not free-swelling balloons) — the ratio range below (0.8, 1.2)
        # reflects that; an earlier (0.2, 2.2) range let untrained A predictions roam far
        # outside physical reality, which is what was blowing up the tube-law residual.
        # P doesn't vary with vessel size nearly as strongly in this
        # tube law, so it keeps a single global scale.
        scales = model_cfg.get("output_scales", {})
        self.register_buffer("area_ratio_scale", torch.tensor(float(scales.get("area_ratio_scale", 0.4))))
        self.register_buffer("area_ratio_min", torch.tensor(float(scales.get("area_ratio_min", 0.8))))
        self.register_buffer("velocity_scale_ms", torch.tensor(float(scales.get("velocity_scale_ms", 0.3))))
        self.register_buffer("flow_ratio_scale", torch.tensor(float(scales.get("flow_ratio_scale", 3.0))))
        # wall_thickness_ratio must match solver.wall_thickness_ratio in the config exactly
        # (see forward() docstring for why P is now computed analytically, not a free head).
        self.register_buffer("wall_thickness_ratio", torch.tensor(float(model_cfg.get("wall_thickness_ratio", 0.1))))

    def _local_area_scale(self, seg_id: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """A0_local (m^2) for each point, read directly from its patient's conditioning
        vector (diameters are stored in cond[:, :n_segments], normalized by /5.0 mm — see
        data/network_topology.py:PatientTopology.conditioning_vector). Keeping this lookup
        self-contained in the model (rather than requiring seg_props to be threaded through
        forward()) means A/Q outputs are always in-scale even outside the training loop
        (e.g. during validation/mra_pipeline.py inference)."""
        diam_norm = cond.gather(1, seg_id.unsqueeze(1))  # (B,1), = diameter_mm / 5.0
        diam_mm = diam_norm * 5.0
        r0_m = (diam_mm / 1000.0) / 2.0
        return math.pi * r0_m ** 2

    def forward(self, x, t, seg_id, cond):
        """
        x, t     : (B, 1) tensors, require_grad=True for PDE residual autograd
        seg_id   : (B,) long tensor, segment index in [0, n_segments)
        cond     : (B, cond_dim) patient conditioning vector

        Returns dict with A, Q, P each (B, 1), in physical units (m^2, m^3/s, Pa).
        A and Q are scaled relative to each point's own segment geometry (see
        _local_area_scale docstring) rather than a fixed global constant.

        P is computed ANALYTICALLY from the predicted A via the same tube law
        data/generate_synthetic.py's Segment1D.pressure() uses — not a free network
        head. In the ground-truth generator, P has no independent degree of freedom:
        it's always P = (4/3)(E h/r0)(sqrt(A/A0)-1), a deterministic function of A.
        An earlier version used a separate head_P trained with a soft tube-law
        residual loss to reconcile it with A; empirically this failed badly (A learned
        the correct pulsatile shape, but P collapsed to near-zero noise, because the
        raw tube-law/momentum residual terms dominated total loss magnitude and won
        the tug-of-war against the comparatively tiny supervised P term, even after
        normalization). Deriving P analytically removes the tug-of-war entirely:
        the tube-law residual is now ~0 by construction, and the supervised P loss
        becomes direct, well-conditioned gradient signal into the A prediction itself.
        """
        seg_emb = self.seg_embedding(seg_id)
        cond_emb = self.graph_encoder(cond)
        h = torch.cat([x, t, seg_emb, cond_emb], dim=-1)
        h = self.trunk(h)
        A_raw = self.head_A(h)
        Q_raw = self.head_Q(h)

        A0_local = self._local_area_scale(seg_id, cond)
        area_ratio = torch.sigmoid(A_raw) * self.area_ratio_scale + self.area_ratio_min  # in (0.8, 1.2)
        A = area_ratio * A0_local

        velocity_char = self.velocity_scale_ms  # m/s, typical mean cerebral arterial velocity
        Q0_local = A0_local * velocity_char
        Q = torch.tanh(Q_raw) * self.flow_ratio_scale * Q0_local

        E_local = cond[:, -1:] * 2.0e6  # undo normalization, see PatientTopology.conditioning_vector
        r0_local = torch.sqrt(A0_local / math.pi)
        h_local = self.wall_thickness_ratio * r0_local
        P = (4.0 / 3.0) * (E_local * h_local / r0_local) * (torch.sqrt(A / A0_local) - 1.0)

        return {"A": A, "Q": Q, "P": P}


def build_model(topology_cfg: dict, model_cfg: dict) -> GPINN:
    n_segments = topology_cfg["n_segments"]
    # cond_dim = diameters (n_segments) + presence mask (n_segments) + stiffness (1)
    cond_dim = 2 * n_segments + 1
    return GPINN(n_segments=n_segments, cond_dim=cond_dim, model_cfg=model_cfg)


if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["topology"], cfg["model"])
    B = 16
    x = torch.rand(B, 1, requires_grad=True)
    t = torch.rand(B, 1, requires_grad=True)
    seg_id = torch.randint(0, cfg["topology"]["n_segments"], (B,))
    cond_dim = 2 * cfg["topology"]["n_segments"] + 1
    cond = torch.rand(B, cond_dim)
    out = model(x, t, seg_id, cond)
    print({k: v.shape for k, v in out.items()})
