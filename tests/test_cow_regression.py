"""The config refactor must not change CoW results: sample 0 of data/cow_v1.h5 (seed 1) is reproduced bit for bit."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from data.generate_cow import run_sample, variants

H5 = Path(__file__).resolve().parents[1] / "data" / "cow_v1.h5"


@pytest.mark.skipif(not H5.exists(), reason="data/cow_v1.h5 not generated")
def test_cow_sample0_unchanged():
    _, _, fields, _ = run_sample((1, 0, variants()))
    with h5py.File(H5, "r") as h:
        g = h["s00000"]
        for v, fd in fields.items():
            for f in "AQP":
                np.testing.assert_array_equal(fd[f], g[f"{v}/{f}"][:])
