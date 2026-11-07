"""Second topology domain (pruned 37-artery network). Needs configs/network_boileau37.yaml, generated from third-party
data by scripts/make_network_boileau37.py (gitignored); skipped when absent."""
from pathlib import Path

import numpy as np
import pytest

from physics.cow_network import build_cow, load_geometry
from physics.solver_1d import simulate_periodic

CFG = Path(__file__).resolve().parents[1] / "configs" / "network_boileau37.yaml"
pytestmark = pytest.mark.skipif(not CFG.exists(), reason="run scripts/make_network_boileau37.py first")


@pytest.fixture(scope="module")
def geo():
    return load_geometry(CFG)


def test_pruning_removes_subtrees(geo):
    _, info = build_cow(("s4",), geo=geo, dx_target=8e-3)
    assert not {"s4", "s5", "s6", "s7"} & set(info["vessels"])
    assert "s3" in info["vessels"] and len(info["vessels"]) == 33


def test_dead_end_variant_rejected(geo):
    with pytest.raises(ValueError):
        build_cow(("s3", "s4"), geo=geo, dx_target=8e-3)


def test_full_network_periodic_balanced_physiological(geo):
    net, info = build_cow((), geo=geo, dx_target=8e-3)
    out = simulate_periodic(net, info["p_init"], n_out=80, tol=1e-5, max_cycles=80, pc_init=info["pc_init"])
    d = out["diagnostics"]
    assert d["converged"] and d["max_courant"] < 1.0
    qin = out["s1"]["Q"][:, 0].mean()
    qout = sum(out[v]["Q"][:, -1].mean() for v in geo["topology"]["outlets"])
    assert qout == pytest.approx(qin, rel=0.01)
    assert qin * 1e6 == pytest.approx(geo["inflow"]["inlet_mean_mL_s"]["s1"], rel=0.01)
    p = np.concatenate([out[v]["P"].ravel() for v in info["vessels"]]) / 133.322
    assert 40 < p.min() and p.max() < 180  # mmHg
