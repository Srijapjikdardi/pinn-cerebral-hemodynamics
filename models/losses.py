"""
Phase 2 — Physics loss engine.

Implements the loss terms from the plan:

    Loss_total = w_data*L_data + w_mass*L_mass + w_momentum*L_momentum
                 + w_bc*L_boundary + w_junction*L_junction

with each PDE residual computed via autograd (torch.autograd.grad) on the
G-PINN's own outputs, evaluated at randomly sampled collocation points
(no ground-truth data needed for these terms — that's the "physics-informed" part).

Self-adaptive weighting: rather than hand-tuning w_1..w_5, we learn per-term
log-variance parameters (Kendall, Gal & Cipolla 2018 "multi-task uncertainty
weighting"), which is a simpler, more stable alternative to the NTK-based
reweighting mentioned in the plan. Set loss_weights.self_adaptive: false in
the config to fall back to fixed weights if you prefer the NTK route later.
"""

import torch
import torch.nn as nn


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(y)/d(x) via autograd, keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def pde_residuals(model, x, t, seg_id, cond, seg_props: dict):
    """
    Evaluate mass + momentum + tube-law residuals at collocation points (x, t).

    seg_props: dict with per-collocation-point tensors:
        rho (B,1), friction_f (B,1), E (B,1), h (B,1), r0 (B,1), A0 (B,1),
        L_m (B,1), T_s (B,1)
    These vary per point because each point belongs to a segment with its own
    geometry/stiffness (pulled from the patient's conditioning info upstream).

    L_m/T_s (segment length in meters, cardiac cycle duration in seconds) correct for
    x, t being sampled normalized to [0,1] rather than physical units: autograd gives
    d(y)/d(x_norm), and chain rule says d(y)/d(x_phys) = d(y)/d(x_norm) * d(x_norm)/d(x_phys)
    = d(y)/d(x_norm) / L_m. Without this, residual magnitude is inconsistently scaled
    across segments of different length/duration — doesn't change the zero-crossing
    (so it doesn't make the physics *wrong*), but biases the optimizer toward whichever
    segments happen to read larger raw residuals. If L_m/T_s are absent from seg_props
    (e.g. older call sites), falls back to uncorrected normalized-coordinate derivatives.
    """
    out = model(x, t, seg_id, cond)
    A, Q, P = out["A"], out["Q"], out["P"]
    L_m = seg_props.get("L_m", 1.0)
    T_s = seg_props.get("T_s", 1.0)

    dA_dt = _grad(A, t) / T_s
    dQ_dx = _grad(Q, x) / L_m
    mass_residual = dA_dt + dQ_dx

    dQ_dt = _grad(Q, t) / T_s
    flux = (Q ** 2) / A
    dFlux_dx = _grad(flux, x) / L_m
    dP_dx = _grad(P, x) / L_m
    # Friction term is f*nu*Q/A (Olufsen et al. 2000 Poiseuille-profile form), not a bare
    # f*Q term — must match data/generate_synthetic.py's _rhs() exactly, or the physics
    # loss the G-PINN is trained against silently diverges from the ground-truth solver.
    momentum_residual = (
        dQ_dt + dFlux_dx + (A / seg_props["rho"]) * dP_dx + seg_props["friction_f"] * Q / A
    )

    # Tube law (constitutive equation) — enforced as a residual too, since P and A
    # are both free network outputs and must stay consistent with each other.
    E, h, r0, A0 = seg_props["E"], seg_props["h"], seg_props["r0"], seg_props["A0"]
    P_tube_law = (4.0 / 3.0) * (E * h / r0) * (torch.sqrt(torch.clamp(A / A0, min=1e-8)) - 1.0)
    tube_law_residual = P - P_tube_law

    return {
        "mass": mass_residual,
        "momentum": momentum_residual,
        "tube_law": tube_law_residual,
        "A": A, "Q": Q, "P": P,
    }


def data_loss(model, x, t, seg_id, cond, A_true, Q_true, P_true, scales: dict):
    """Supervised loss against ground truth. Normalized per-field by a characteristic
    scale before averaging — without this, raw MSE is dominated entirely by pressure
    (P ~ 1e3 Pa, so (P_err)^2 ~ 1e6) while area error (A ~ 1e-6 m^2, (A_err)^2 ~ 1e-12)
    contributes essentially nothing to the gradient. `scales` should have keys
    'A_scale', 'Q_scale', 'P_scale' (e.g. the same order-of-magnitude values used for
    model output scaling, or per-batch characteristic values — either works as long as
    it's consistent)."""
    out = model(x, t, seg_id, cond)
    A_err = (out["A"] - A_true) / scales["A_scale"]
    Q_err = (out["Q"] - Q_true) / scales["Q_scale"]
    P_err = (out["P"] - P_true) / scales["P_scale"]
    return (A_err ** 2).mean() + (Q_err ** 2).mean() + (P_err ** 2).mean()


def boundary_loss(model, x_bc, t_bc, seg_id_bc, cond_bc, Q_bc_target, Q_scale: float = 1.5e-6):
    """Inlet/outlet boundary conditions (e.g. prescribed ICA inflow waveform).
    Normalized by Q_scale for the same reason data_loss normalizes per-field — raw Q
    is ~1e-6 (m^3/s), so unnormalized MSE is ~1e-12, which would force self-adaptive
    weighting to compensate for a six-order-of-magnitude scale gap rather than for
    genuine term difficulty."""
    out = model(x_bc, t_bc, seg_id_bc, cond_bc)
    return nn.functional.mse_loss(out["Q"] / Q_scale, Q_bc_target / Q_scale)


def junction_loss(model, junction_points: list, cond, seg_id_lookup: dict,
                   Q_scale: float = 1.5e-6, P_scale: float = 5000.0):
    """
    Kirchhoff constraints at each arterial junction node:
        sum(Q_in) == sum(Q_out)   (mass conservation at the branch point)
        P equal across all segments meeting at that node

    junction_points: list of dicts, one per junction, each containing:
        {"segments": [seg_name, ...], "signs": [+1/-1, ...],
         "x": [x_at_node_for_each_seg (B,1)], "t": (B,1) shared time,
         "seg_ids": [(B,) long tensor per segment]}
    cond: (B, cond_dim) shared patient conditioning vector for this batch

    Both terms normalized by characteristic scale before squaring — same reason
    data_loss normalizes per-field. Confirmed by direct measurement: unnormalized,
    raw Q (~1e-6, squared ~1e-12) was completely swamped by raw P (~1e3 Pa, squared
    ~1e6), making this loss effectively pressure-only and, combined with its already
    large raw magnitude, dominant enough to collapse training toward a trivial
    steady-state solution that ignored the real supervised pulsatile target.

    Returns scalar loss combining mass-balance residual and pressure-mismatch residual,
    averaged over all sampled junctions.
    """
    total_mass_res = 0.0
    total_pressure_res = 0.0
    n = 0
    for jp in junction_points:
        Qs, Ps = [], []
        for x_i, seg_id_i, sign in zip(jp["x"], jp["seg_ids"], jp["signs"]):
            out = model(x_i, jp["t"], seg_id_i, cond)
            Qs.append(sign * out["Q"] / Q_scale)
            Ps.append(out["P"] / P_scale)
        q_sum = torch.stack(Qs, dim=0).sum(dim=0)
        total_mass_res = total_mass_res + (q_sum ** 2).mean()

        p_stack = torch.stack(Ps, dim=0)  # (n_segs_at_node, B, 1)
        p_mean = p_stack.mean(dim=0, keepdim=True)
        total_pressure_res = total_pressure_res + ((p_stack - p_mean) ** 2).mean()
        n += 1

    if n == 0:
        return torch.tensor(0.0)
    return total_mass_res / n + total_pressure_res / n


class SelfAdaptiveWeights(nn.Module):
    """Learnable per-term log-variance weights (Kendall et al. 2018 style):
        L_total = sum_i [ exp(-log_var_i) * L_i + log_var_i ]
    This lets the optimizer automatically down-weight noisy/harder-to-fit terms
    instead of you hand-tuning w_1..w_5 in the config."""

    def __init__(self, term_names):
        super().__init__()
        self.term_names = term_names
        self.log_vars = nn.Parameter(torch.zeros(len(term_names)))

    def forward(self, losses: dict):
        total = 0.0
        breakdown = {}
        for i, name in enumerate(self.term_names):
            if name not in losses:
                continue
            lv = self.log_vars[i]
            weighted = torch.exp(-lv) * losses[name] + lv
            breakdown[name] = weighted.item() if hasattr(weighted, "item") else weighted
            total = total + weighted
        return total, breakdown
