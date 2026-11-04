"""Gate 0 / physics checks on the bundled solver (data/generate_synthetic.py).

xfail(strict=True) = known defect documented in docs/audit.md; remove the marker when fixed.
"""
import numpy as np
import pytest

from data.generate_synthetic import Segment1D, local_fd_solver, physiological_ica_waveform
from data.network_topology import junction_map

from .conftest import make_patient  # noqa: F401  (importlib mode: relative import of conftest)


INLET_SEGMENTS = ("ICA_L", "ICA_R", "PCA_P1_L", "PCA_P1_R")  # x=0 of these carries prescribed inflow


def _volume_budget(out):
    t = out["ICA_L"]["t"]
    inflow = sum(out[s]["Q"][:, 0] for s in INLET_SEGMENTS if s in out)
    outflow = sum(out[s]["Q"][:, -1] for s in ("MCA_L", "MCA_R") if s in out)
    stored = np.zeros_like(t)
    for sd in out.values():
        dx = sd["x"][1] - sd["x"][0]
        stored += np.trapezoid(sd["A"], dx=dx, axis=1)
    v_in = float(np.trapezoid(inflow, t))
    v_out = float(np.trapezoid(outflow, t))
    d_stored = float(stored[-1] - stored[0])
    return v_in, v_out, d_stored


def test_solution_is_finite_with_positive_area(solved):
    out, _ = solved
    for name, sd in out.items():
        for f in ("A", "Q", "P"):
            assert np.isfinite(sd[f]).all(), (name, f)
        assert (sd["A"] > 0).all(), name


def test_inlet_flow_matches_prescribed_waveform(solved):
    out, sweep = solved
    t = out["ICA_L"]["t"]
    dt = t[0]  # first saved sample is after step 0, so t[0] == dt
    # solver evaluates the waveform at the step's start time, then increments t before saving
    expected = physiological_ica_waveform(t - dt, sweep["cardiac_cycle_s"]) * 1e-6 / 2.0
    np.testing.assert_allclose(out["ICA_L"]["Q"][:, 0], expected, rtol=1e-6, atol=1e-12)


def test_junction_boundary_values_satisfy_kirchhoff(solved):
    """Passes, but note it is satisfied by construction (the solver overwrites junction
    boundary values to enforce it), so it says nothing about interior mass conservation."""
    out, _ = solved
    q_scale = max(np.abs(sd["Q"]).max() for sd in out.values())
    for node, touching in junction_map().items():
        touching = [(s, sg) for s, sg in touching if s in out]
        if node == "BA" or len(touching) < 2:
            continue
        resid = sum(sg * (out[s]["Q"][:, -1] if sg == 1 else out[s]["Q"][:, 0]) for s, sg in touching)
        assert np.abs(resid).max() < 1e-9 * q_scale, node


def test_solver_is_deterministic(cfg, solved):
    out, sweep = solved
    again = local_fd_solver(make_patient(), sweep, cfg["solver"])
    np.testing.assert_array_equal(out["MCA_L"]["Q"], again["MCA_L"]["Q"])


@pytest.mark.xfail(
    strict=True,
    reason="Network volume balance fails: ~99.5% of imposed inflow volume is neither stored nor discharged",
)
def test_global_volume_balance(solved):
    """int(Q_in) - int(Q_out) = change in stored volume, within 5% of the inflow volume."""
    out, _ = solved
    v_in, v_out, d_stored = _volume_budget(out)
    assert v_in > 0
    assert abs(v_in - v_out - d_stored) / v_in < 0.05


@pytest.mark.xfail(
    strict=True,
    reason="Distal ends of MCA (and other terminals) are never updated: Q stays 0, no outlet BC exists",
)
def test_terminal_ends_carry_flow(solved):
    out, _ = solved
    v_in, v_out, _ = _volume_budget(out)
    assert v_out > 0.10 * v_in


@pytest.mark.xfail(
    strict=True,
    reason="MCA segment: ~1 mL enters at x=0, 0 leaves at x=L, ~0.001 mL stored",
)
def test_mca_segment_volume_budget(solved):
    out, _ = solved
    sd = out["MCA_L"]
    t = sd["t"]
    dx = sd["x"][1] - sd["x"][0]
    q_in = np.trapezoid(sd["Q"][:, 0], t)
    q_out = np.trapezoid(sd["Q"][:, -1], t)
    vol = np.trapezoid(sd["A"], dx=dx, axis=1)
    assert abs(q_in - q_out - (vol[-1] - vol[0])) / abs(q_in) < 0.05


@pytest.mark.xfail(
    strict=True,
    reason="Segment1D.wave_speed() = sqrt(E h/(2 rho r0)); the tube law used by the solver implies "
           "sqrt(2 E h/(3 rho r0)). Ratio 0.866: CFL step is based on a wave speed 13% too low",
)
@pytest.mark.parametrize("E", [4.0e5, 8.0e5, 2.0e6])
@pytest.mark.parametrize("d_mm", [1.4, 3.0, 4.5])
def test_wave_speed_estimate_matches_tube_law(E, d_mm):
    seg = Segment1D("x", 0.015, d_mm / 2000.0, E, 0.1, 6, 1060.0, 22.0, 3.5e-3)
    A0 = seg.A0
    eps = 1e-6 * A0
    dPdA = (seg.pressure(np.array([A0 + eps]))[0] - seg.pressure(np.array([A0 - eps]))[0]) / (2 * eps)
    c_tube_law = np.sqrt(A0 / 1060.0 * dPdA)
    assert seg.wave_speed() == pytest.approx(c_tube_law, rel=0.02)
