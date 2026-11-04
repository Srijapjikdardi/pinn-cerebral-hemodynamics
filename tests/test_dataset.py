"""Gate 0 / contract checks on the shipped dataset data/synthetic_demo.h5 (60 patients)."""
import h5py
import numpy as np
import pytest

from data.network_topology import SEGMENTS

from .conftest import DEMO_H5

pytestmark = pytest.mark.skipif(not DEMO_H5.exists(), reason="shipped demo dataset not present")


def _patient(h5, key):
    g = h5[key]
    segs = {sk[len("segment_"):]: {f: g[sk][f][:] for f in g[sk].keys()}
            for sk in g.keys() if sk.startswith("segment_")}
    return g, segs


@pytest.fixture(scope="module")
def h5():
    with h5py.File(DEMO_H5, "r") as f:
        yield f


def test_no_nan_or_inf(h5):
    for key in sorted(h5.keys()):
        _, segs = _patient(h5, key)
        for name, sd in segs.items():
            for f in ("A", "Q", "P"):
                assert np.isfinite(sd[f]).all(), (key, name, f)


def test_conditioning_vector_length(h5):
    expected = 2 * len(SEGMENTS) + 1
    for key in sorted(h5.keys()):
        assert h5[key]["conditioning"]["vector"].shape == (expected,)


def test_area_ratio_inside_model_output_window(h5):
    """models/gpinn.py can only output A/A0 in (0.8, 1.2); data outside would be unlearnable."""
    d_norm_index = {s[0]: i for i, s in enumerate(SEGMENTS)}
    for key in sorted(h5.keys()):
        g, segs = _patient(h5, key)
        cond = g["conditioning"]["vector"][:]
        for name, sd in segs.items():
            d_mm = cond[d_norm_index[name]] * 5.0
            a0 = np.pi * (d_mm / 2000.0) ** 2
            ratio = sd["A"] / a0
            assert ratio.min() > 0.8 and ratio.max() < 1.2, (key, name, ratio.min(), ratio.max())


@pytest.mark.xfail(
    strict=True,
    reason="Shipped ground truth does not conserve volume: stored volume change is ~0.5% of inflow",
)
@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_dataset_volume_balance(h5, index):
    key = sorted(h5.keys())[index]
    _, segs = _patient(h5, key)
    t = segs["ICA_L"]["t"]
    inflow = sum(segs[s]["Q"][:, 0] for s in ("ICA_L", "ICA_R", "PCA_P1_L", "PCA_P1_R"))
    outflow = sum(segs[s]["Q"][:, -1] for s in ("MCA_L", "MCA_R") if s in segs)
    stored = sum(np.trapezoid(sd["A"], dx=sd["x"][1] - sd["x"][0], axis=1) for sd in segs.values())
    v_in = np.trapezoid(inflow, t)
    v_out = np.trapezoid(outflow, t)
    assert abs(v_in - v_out - (stored[-1] - stored[0])) / v_in < 0.05
