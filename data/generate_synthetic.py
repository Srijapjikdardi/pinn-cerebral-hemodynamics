"""
Phase 1 — Synthetic "ground truth" data generation.

Runs a parametric sweep of N synthetic CoW patients (randomized stiffness +
anatomical variation, per configs/default.yaml) and solves the 1D pulsatile
blood-flow equations for each, exporting (x, t, A, Q, P) per segment to HDF5.

SOLVER NOTE
-----------
The plan calls for OpenCOR / SimVascular. Those are excellent but heavy
external dependencies (GUI installers, XML config formats) that don't run
cleanly in a lightweight sandbox. This file ships a self-contained explicit
finite-difference solver (`solvers.local_fd_solver`) for the *exact same*
1D elastic-tube PDE system the G-PINN is trained against:

    Mass:      dA/dt + dQ/dx = 0
    Momentum:  dQ/dt + d/dx(Q^2/A) + (A/rho) dP/dx + f*Q = 0
    Tube law:  P - P0 = (4/3)(E*h/r0)(sqrt(A/A0) - 1)

Per-segment PDEs are solved with a MacCormack predictor-corrector scheme (with a
CFL-based adaptive timestep); segments are coupled at junction nodes each timestep
via a fixed-point (Gauss-Seidel) relaxation enforcing (a) pressure continuity and
(b) Kirchhoff mass balance at every node — same physics as models/losses.py enforces
on the network side.

To swap in SimVascular for clinical-grade ground truth: implement
`solvers.simvascular_solver(patient, sweep_cfg)` with the same return
signature as `local_fd_solver` and set `solver.type: simvascular` in the config.
"""

import argparse
import yaml
import numpy as np
import h5py
from tqdm import tqdm

from data.network_topology import SEGMENTS, make_random_patient, junction_map, PatientTopology


# ------------------------------------------------------------------
# Inlet waveform
# ------------------------------------------------------------------
def physiological_ica_waveform(t: np.ndarray, cycle_s: float) -> np.ndarray:
    """Parametric pulsatile flow waveform (mL/s) approximating an ICA inlet trace:
    fast systolic upstroke, dicrotic notch, diastolic decay. Sum-of-Gaussians
    parametrization — swap for a digitized real waveform if you have one."""
    phase = np.mod(t, cycle_s) / cycle_s
    systolic = 5.0 * np.exp(-0.5 * ((phase - 0.15) / 0.06) ** 2)
    dicrotic = 1.2 * np.exp(-0.5 * ((phase - 0.40) / 0.05) ** 2)
    baseline = 2.0
    return baseline + systolic + dicrotic  # mL/s


# ------------------------------------------------------------------
# Local (bundled) 1D elastic-tube solver
# ------------------------------------------------------------------
def _d_forward(f, dx):
    """One-sided forward difference, edge-replicated at the last point."""
    d = np.empty_like(f)
    d[:-1] = (f[1:] - f[:-1]) / dx
    d[-1] = d[-2]
    return d


def _d_backward(f, dx):
    """One-sided backward difference, edge-replicated at the first point."""
    d = np.empty_like(f)
    d[1:] = (f[1:] - f[:-1]) / dx
    d[0] = d[1]
    return d


class Segment1D:
    def __init__(self, name, length_m, r0_m, E_pa, h_ratio, n_x, rho, friction_f, viscosity_pa_s):
        self.name = name
        self.L = length_m
        self.r0 = r0_m
        self.A0 = np.pi * r0_m ** 2
        self.E = E_pa
        self.h = h_ratio * r0_m
        self.n_x = n_x
        self.dx = length_m / (n_x - 1)
        self.rho = rho
        self.f = friction_f
        self.nu = viscosity_pa_s / rho  # kinematic viscosity
        self.A = np.full(n_x, self.A0)
        self.Q = np.zeros(n_x)

    def pressure(self, A):
        # Tube law: P - P0 = (4/3)(E h / r0)(sqrt(A/A0) - 1), P0 = 0 gauge pressure
        return (4.0 / 3.0) * (self.E * self.h / self.r0) * (np.sqrt(np.maximum(A, 1e-12) / self.A0) - 1.0)

    def wave_speed(self) -> float:
        """Moens-Korteweg pulse-wave speed c = sqrt(E*h / (2*rho*r0)), used only to pick
        a CFL-stable dt. This is an estimate (assumes small perturbations about A0), not
        used anywhere in the actual residual/flux computation."""
        return float(np.sqrt(self.E * self.h / (2.0 * self.rho * self.r0)))

    def max_stable_dt(self, cfl_safety=0.8) -> float:
        c = self.wave_speed()
        return cfl_safety * self.dx / max(c, 1e-6)

    def _rhs(self, A, Q, diff):
        """Right-hand-side residual (mass_rate, momentum_rate) using the given
        differencing operator (forward for the predictor stage, backward for
        the corrector stage) — this upwind-biasing is what makes MacCormack
        stable where plain centered differencing (FTCS) is not."""
        P = self.pressure(A)
        flux_mom = Q ** 2 / np.maximum(A, 1e-12)
        dQdx = diff(Q, self.dx)
        dFluxdx = diff(flux_mom, self.dx)
        dPdx = diff(P, self.dx)
        # Viscous (Poiseuille-profile) friction term: f * nu * Q / A — see Olufsen et al.
        # (2000). Using a bare f*Q here (without the nu/A factor) over-damps flow by several
        # orders of magnitude, since f~22 is meant to multiply kinematic viscosity (~1e-6),
        # not stand alone.
        friction_term = self.f * self.nu * Q / np.maximum(A, 1e-12)
        dA = -dQdx
        dQ = -(dFluxdx + (A / self.rho) * dPdx + friction_term)
        return dA, dQ

    def step(self, dt):
        """MacCormack predictor-corrector step on interior points.
        Boundary points are overwritten by junction/BC coupling after this call.
        Requires dt <= max_stable_dt() for the CFL condition to hold."""
        A, Q = self.A, self.Q

        # Predictor (forward differences)
        dA1, dQ1 = self._rhs(A, Q, _d_forward)
        A_p = A + dt * dA1
        Q_p = Q + dt * dQ1
        A_p = np.clip(A_p, 0.05 * self.A0, 20.0 * self.A0)  # guard against transient blow-up

        # Corrector (backward differences on the predicted state)
        dA2, dQ2 = self._rhs(A_p, Q_p, _d_backward)
        A_c = A_p + dt * dA2
        Q_c = Q_p + dt * dQ2

        A_new = 0.5 * (A + A_c)
        Q_new = 0.5 * (Q + Q_c)
        A_new = np.clip(A_new, 0.05 * self.A0, 20.0 * self.A0)

        self.A[1:-1] = A_new[1:-1]
        self.Q[1:-1] = Q_new[1:-1]


def local_fd_solver(patient: PatientTopology, sweep_cfg: dict, solver_cfg: dict):
    """Solve the network for one patient. Returns dict: segment_name -> {x, t, A, Q, P}."""
    rho = solver_cfg["blood_density_kg_m3"]
    friction_f = solver_cfg["friction_coefficient"]
    viscosity_pa_s = solver_cfg["blood_dynamic_viscosity_pa_s"]
    h_ratio = solver_cfg["wall_thickness_ratio"]
    n_x = sweep_cfg["spatial_points_per_segment"]
    cycle_s = sweep_cfg["cardiac_cycle_s"]
    n_cycles = sweep_cfg["n_cycles_simulated"]
    total_t = cycle_s * n_cycles

    segments = {}
    for name, start, end, _, _ in SEGMENTS:
        d_mm = patient.diameters_mm[name]
        if d_mm <= 1e-6:
            continue  # vessel absent for this patient
        r0_m = (d_mm / 1000.0) / 2.0
        length_m = patient.segment_lengths_mm[name] / 1000.0
        segments[name] = Segment1D(name, length_m, r0_m, patient.stiffness_E_pa,
                                    h_ratio, n_x, rho, friction_f, viscosity_pa_s)

    # CFL condition: dt must be <= dx/wave_speed for every segment, or the explicit
    # scheme diverges. Take the tightest constraint across the network and cap it
    # at the config value (which acts as an upper bound / sanity ceiling).
    cfl_dt = min(seg.max_stable_dt() for seg in segments.values())
    dt = min(cfl_dt, solver_cfg["dt_s"])

    jmap = junction_map()
    n_steps = int(total_t / dt)
    save_every = max(1, n_steps // (sweep_cfg["time_points_per_cycle"] * n_cycles))

    history = {name: {"A": [], "Q": [], "P": [], "t": []} for name in segments}
    t = 0.0
    for step in range(n_steps):
        # 1) advance each segment's interior
        for seg in segments.values():
            seg.step(dt)

        # 2) inlet BC: ICA_L / ICA_R driven by the pulsatile waveform (split evenly
        #    if both present; full flow to whichever is present if only one is)
        q_in = physiological_ica_waveform(np.array([t]), cycle_s)[0] * 1e-6  # mL/s -> m^3/s
        inlets = [s for s in ("ICA_L", "ICA_R") if s in segments]
        for s in inlets:
            segments[s].Q[0] = q_in / len(inlets)
            segments[s].A[0] = segments[s].A[1]  # zero-gradient area at inlet

        # BA (basilar) inlet: simplified constant vertebrobasilar contribution
        if "PCA_P1_L" in segments:
            segments["PCA_P1_L"].Q[0] = 0.5 * q_in * 0.3
        if "PCA_P1_R" in segments:
            segments["PCA_P1_R"].Q[0] = 0.5 * q_in * 0.3

        # 3) junction coupling: enforce pressure continuity + Kirchhoff mass balance
        #    via a few Gauss-Seidel relaxation sweeps per timestep
        #
        #    IMPORTANT: Segment.step() only advances *interior* points (indices
        #    1..-2); the junction-adjacent boundary points are never touched by
        #    the interior scheme. Without an explicit extrapolation step here,
        #    those boundary points stay frozen at their initial values forever
        #    and no flow/pressure ever reaches downstream segments. So: first
        #    extrapolate each segment's junction-end value from its nearest
        #    interior neighbor (a standard zero-gradient / characteristic-based
        #    outflow approximation), *then* apply the Kirchhoff/pressure
        #    relaxation on top of that physically-updated starting point.
        # BA is a prescribed inlet (both PCA segments originate there from the
        # vertebrobasilar supply, which we model as an external BC, not as its
        # own segment) — exclude it from generic conservation coupling, or the
        # extrapolation step below would immediately overwrite the inlet BC
        # set in step 2 above.
        # A "node" with fewer than 2 segments touching it is an inlet or a terminal
        # end (e.g. ICA_L's own start point, or an MCA's distal end), not an actual
        # branch point — it must be left alone here, or this loop will stomp on
        # inlet BCs set in step 2 (this was the previous bug: 'ICA_L' node has a
        # single touching segment, so "extrapolating" it just overwrote the BC
        # with itself one index over).
        real_junction_nodes = {n: t for n, t in jmap.items() if n != "BA" and len(t) >= 2}

        for node, touching in real_junction_nodes.items():
            for seg_name, sign in touching:
                if seg_name not in segments:
                    continue
                seg = segments[seg_name]
                idx, neighbor = (-1, -2) if sign == +1 else (0, 1)
                seg.A[idx] = seg.A[neighbor]
                seg.Q[idx] = seg.Q[neighbor]

        for _ in range(3):
            for node, touching in real_junction_nodes.items():
                touching = [(s, sign) for s, sign in touching if s in segments]
                if len(touching) < 2:
                    continue
                # average pressure at the shared node -> assign to each segment's
                # node-adjacent boundary area via inverse tube law
                p_vals = []
                for seg_name, sign in touching:
                    seg = segments[seg_name]
                    idx = -1 if sign == +1 else 0
                    p_vals.append(seg.pressure(np.array([seg.A[idx]]))[0])
                p_avg = float(np.mean(p_vals))
                for seg_name, sign in touching:
                    seg = segments[seg_name]
                    idx = -1 if sign == +1 else 0
                    target_ratio = 1.0 + p_avg / ((4.0 / 3.0) * (seg.E * seg.h / seg.r0))
                    seg.A[idx] = seg.A0 * max(target_ratio, 0.2) ** 2

                # Kirchhoff: sum(Q_in) == sum(Q_out) at this node, distribute residual
                q_sum = sum(sign * segments[s].Q[-1 if sign == 1 else 0] for s, sign in touching)
                n_touch = len(touching)
                for seg_name, sign in touching:
                    seg = segments[seg_name]
                    idx = -1 if sign == +1 else 0
                    seg.Q[idx] -= sign * q_sum / n_touch

        t += dt
        if step % save_every == 0 and t > cycle_s * (n_cycles - 1):  # keep last cycle only
            for name, seg in segments.items():
                history[name]["A"].append(seg.A.copy())
                history[name]["Q"].append(seg.Q.copy())
                history[name]["P"].append(seg.pressure(seg.A))
                history[name]["t"].append(t)

    out = {}
    for name, seg in segments.items():
        out[name] = {
            "x": np.linspace(0, seg.L, seg.n_x),
            "t": np.array(history[name]["t"]),
            "A": np.array(history[name]["A"]),
            "Q": np.array(history[name]["Q"]),
            "P": np.array(history[name]["P"]),
        }
    return out


# ------------------------------------------------------------------
# Sweep driver
# ------------------------------------------------------------------
def run_sweep(config_path: str, out_path: str, start_idx: int = 0, end_idx: int = None):
    """
    start_idx/end_idx let you generate a dataset in chunks across multiple calls
    (useful when a single call's time budget can't cover n_simulations patients).
    rng is advanced past [0, start_idx) patients first so the sequence of patients
    generated is IDENTICAL to what a single uninterrupted call would produce —
    chunking doesn't change which patients you get, only how many calls it takes.
    Appends to out_path if it already exists (mode "a"), else creates it (mode "w").
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["sweep"]["random_seed"])
    n_sims = cfg["sweep"]["n_simulations"]
    end_idx = end_idx if end_idx is not None else n_sims

    for _ in range(start_idx):
        make_random_patient(rng, cfg["sweep"])  # discard, just advancing rng to stay in sequence

    import time as _time
    import os as _os

    mode = "a" if _os.path.exists(out_path) else "w"
    with h5py.File(out_path, mode) as h5:
        for i in tqdm(range(start_idx, end_idx), desc="Simulating synthetic patients"):
            patient = make_random_patient(rng, cfg["sweep"])
            t0 = _time.time()
            if cfg["solver"]["type"] == "local_fd":
                result = local_fd_solver(patient, cfg["sweep"], cfg["solver"])
            else:
                raise NotImplementedError(
                    "solver.type='simvascular' is a stub — implement "
                    "solvers.simvascular_solver() with the same signature as "
                    "local_fd_solver() to swap in higher-fidelity ground truth."
                )
            solve_time_s = _time.time() - t0

            grp = h5.create_group(f"patient_{i:05d}")
            grp.attrs["stiffness_E_pa"] = patient.stiffness_E_pa
            grp.attrs["solver_time_s"] = solve_time_s  # used by validation/evaluate.py Phase 4.2
            cond_grp = grp.create_group("conditioning")
            cond_grp.create_dataset("vector", data=patient.conditioning_vector())
            for seg_name, seg_data in result.items():
                sg = grp.create_group(f"segment_{seg_name}")
                for k, v in seg_data.items():
                    sg.create_dataset(k, data=v)

    print(f"Wrote patients [{start_idx}, {end_idx}) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/synthetic_dataset.h5")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()
    run_sweep(args.config, args.out, args.start_idx, args.end_idx)
