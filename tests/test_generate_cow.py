import json

import h5py
import numpy as np

from data.generate_cow import NT, NX, run_sample, sample_params, variants


def test_variant_count():
    vl = variants()
    assert len(vl) == 96 and len(set(vl)) == 96


def test_sampling_deterministic():
    vl = variants()
    assert sample_params(3, 7, vl) == sample_params(3, 7, vl)
    assert sample_params(3, 7, vl) != sample_params(3, 8, vl)


def test_sample_shapes_converged_balanced():
    vl = variants()
    idx, meta, fields, t = run_sample((0, 0, vl))
    assert meta["converged"] and meta["max_courant"] < 1.0
    for n, fd in fields.items():
        for f in ("A", "Q", "P"):
            assert fd[f].shape == (NT, NX) and np.isfinite(fd[f]).all()
        assert (fd["A"] > 0).all()
    ins = [n for n in ("ICA_L", "ICA_R", "BA") if n in fields]
    outs = [n for n in ("MCA_L", "MCA_R", "A2_L", "A2_R", "P2_L", "P2_R") if n in fields]
    qin = sum(fields[n]["Q"][:, 0].mean() for n in ins)
    qout = sum(fields[n]["Q"][:, -1].mean() for n in outs)
    assert abs(qin - qout) / qin < 5e-3
