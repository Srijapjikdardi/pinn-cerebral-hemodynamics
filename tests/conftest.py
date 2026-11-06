"""Shared fixtures for Gate 0 tests.

The solver tests use a shortened cardiac cycle (0.2 s) so the full suite stays fast
(a few seconds per solve instead of ~11 s). The conservation failures they expose do
not depend on cycle length.
"""
import copy
from pathlib import Path

import pytest
import yaml

from data.generate_synthetic import local_fd_solver
from data.network_topology import SEGMENTS, PatientTopology, default_segment_lengths_mm

ROOT = Path(__file__).resolve().parents[1]
DEMO_H5 = ROOT / "data" / "synthetic_demo.h5"


def make_patient(absent=(), E=8.0e5):
    """Reference patient: baseline diameters, default lengths, optional absent segments."""
    diameters = {s[0]: (0.0 if s[0] in absent else s[3]) for s in SEGMENTS}
    return PatientTopology(
        diameters_mm=diameters,
        stiffness_E_pa=E,
        segment_lengths_mm=default_segment_lengths_mm(),
    )


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())


@pytest.fixture(scope="session")
def solved(cfg):
    """One solve of the complete CoW, shared by all solver tests."""
    sweep = copy.deepcopy(cfg["sweep"])
    sweep["cardiac_cycle_s"] = 0.2
    sweep["n_cycles_simulated"] = 1
    out = local_fd_solver(make_patient(), sweep, cfg["solver"])
    return out, sweep


@pytest.fixture(scope="session")
def cow_pilot(tmp_path_factory):
    """Small verified-solver dataset (8 samples, seed 7) generated on demand, so tests do not depend on gitignored files."""
    import subprocess
    import sys
    out = tmp_path_factory.mktemp("cow") / "cow_pilot.h5"
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_cow.py"), "--n", "8", "--out", str(out), "--seed", "7",
                    "--workers", "1"], check=True, cwd=ROOT, capture_output=True)
    return out
