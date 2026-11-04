import numpy as np
import pytest

from physics.cow_network import build_cow, load_geometry
from physics.solver_1d import simulate_periodic


def _run(absent=(), dx=2e-3, cycles=40):
    net, info = build_cow(absent, dx_target=dx)
    out = simulate_periodic(net, info["p_init"], max_cycles=cycles, tol=1e-5, n_out=80)
    return net, out


def _mean_end_flow(out, name, end):
    return out[name]["Q"][:, 0 if end == 0 else -1].mean()


@pytest.fixture(scope="module")
def full():
    return _run(())


def test_full_network_converges_and_balances(full):
    net, out = full
    d = out["diagnostics"]
    assert d["converged"], d
    assert (d["cycles_to_converge"], d["max_courant"] < 1.0) == (d["cycles_to_converge"], True)
    # periodic state: mean inflow equals mean outflow over the recorded cycle
    qin = sum(_mean_end_flow(out, n, 0) for n in ("ICA_L", "ICA_R", "BA"))
    qout = sum(_mean_end_flow(out, n, 1) for n in ("MCA_L", "MCA_R", "A2_L", "A2_R", "P2_L", "P2_R"))
    assert qout == pytest.approx(qin, rel=0.01)


def test_pressure_physiological(full):
    _, out = full
    p = out["MCA_L"]["P"] / 133.322
    assert 60 < p.min() and p.max() < 220


def test_variants_are_real_graph_changes():
    net_a, info_a = build_cow(("A1_L", "ACoA"))
    assert "A2_L" not in info_a["vessels"]  # no flow path: pruned
    net_b, info_b = build_cow(("A1_L",))
    assert "A2_L" in info_b["vessels"]  # fed through ACoA
    with pytest.raises(ValueError):
        build_cow(("ICA_L",))


def test_acoa_carries_flow_when_a1_absent():
    full_net, full_out = _run(())
    _, out = _run(("A1_L",))
    q_full = np.abs(out["ACoA"]["Q"]).mean()
    q_ref = np.abs(full_out["ACoA"]["Q"]).mean()
    assert q_full > 20 * max(q_ref, 1e-12)  # ACoA becomes the feeder of the left ACA


def test_pcoa_absent_changes_pca_waveform():
    """Mean PCA flow is set by the terminal resistance, so compare the waveform, not the mean."""
    _, out0 = _run(())
    _, out1 = _run(("PCoA_L",))
    a, b = out0["P2_L"]["Q"][:, 0], out1["P2_L"]["Q"][:, 0]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) > 1e-3
