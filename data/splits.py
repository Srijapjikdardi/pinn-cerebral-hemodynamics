"""Leave-topology-out splits over the CoW dataset (data/generate_cow.py).

A sample's topology = frozenset of absent variable vessels (96 valid variants). Splits are defined on variants so no
held-out topology ever appears in training. Samples are indexed by their position in sorted group order.

split names:
  random      : sample-level 70/15/15 (in-distribution baseline; every topology seen)
  variant     : hold out a random 25% of variants (seeded) as test_ood; val = unseen physiology of train variants
  complexity  : train/val on <=2 absent vessels, test_ood on >=4 absent (extrapolation in topology complexity)
  vessel_<V>  : hold out every variant in which vessel V is absent (leave-one-feature-out), V in variable_vessels
Each returns dict(train, val, test_id, test_ood) of sample indices; test_id = unseen physiology on seen topologies.
"""
import json

import h5py
import numpy as np


def load_topologies(h5_path):
    with h5py.File(h5_path, "r") as h5:
        keys = sorted(k for k in h5.keys())
        return [tuple(sorted(json.loads(h5[k].attrs["meta"])["absent"])) for k in keys]


def _by_group(topos, in_set):
    return np.array([i for i, t in enumerate(topos) if in_set(t)], dtype=int)


def make_split(topos, name, seed=0, test_frac=0.15, val_frac=0.15):
    rng = np.random.default_rng(seed)
    n = len(topos)
    allv = sorted(set(topos))
    if name == "random":
        p = rng.permutation(n)
        a, b = int(n * (1 - test_frac - val_frac)), int(n * (1 - test_frac))
        return dict(train=np.sort(p[:a]), val=np.sort(p[a:b]), test_id=np.sort(p[b:]), test_ood=np.array([], int))
    if name == "variant":
        k = max(1, int(round(0.25 * len(allv))))
        held = {allv[i] for i in rng.choice(len(allv), size=k, replace=False)}
        ood = _by_group(topos, lambda t: t in held)
        seen = _by_group(topos, lambda t: t not in held)
    elif name == "complexity":
        ood = _by_group(topos, lambda t: len(t) >= 4)
        seen = _by_group(topos, lambda t: len(t) <= 2)
    elif name.startswith("vessel_"):
        v = name[len("vessel_"):]
        ood = _by_group(topos, lambda t: v in t)
        seen = _by_group(topos, lambda t: v not in t)
    else:
        raise ValueError(name)
    p = rng.permutation(seen)
    nt, nv = int(len(p) * test_frac), int(len(p) * val_frac)
    return dict(train=np.sort(p[nt + nv:]), val=np.sort(p[nt:nt + nv]), test_id=np.sort(p[:nt]), test_ood=ood)


def describe(topos, sp):
    return {k: dict(n=len(v), variants=len({topos[i] for i in v})) for k, v in sp.items()}
