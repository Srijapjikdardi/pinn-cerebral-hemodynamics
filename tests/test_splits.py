import itertools

import numpy as np
import pytest

from data.splits import make_split

VV = ["ACoA", "PCoA_L", "PCoA_R", "A1_L", "A1_R", "P1_L", "P1_R"]
TOPOS = [tuple(sorted(c)) for k in range(5) for c in itertools.combinations(VV, k)] * 3


@pytest.mark.parametrize("name", ["variant", "complexity", "vessel_PCoA_L", "vessel_ACoA"])
def test_no_topology_leak(name):
    sp = make_split(TOPOS, name)
    tr = {TOPOS[i] for i in sp["train"]} | {TOPOS[i] for i in sp["val"]}
    ood = {TOPOS[i] for i in sp["test_ood"]}
    assert ood and not (tr & ood)
    assert not (set(sp["train"]) & (set(sp["val"]) | set(sp["test_id"]) | set(sp["test_ood"])))
    assert {TOPOS[i] for i in sp["test_id"]} <= tr | {TOPOS[i] for i in sp["test_id"]}


def test_vessel_split_semantics():
    sp = make_split(TOPOS, "vessel_PCoA_L")
    assert all("PCoA_L" in TOPOS[i] for i in sp["test_ood"])
    assert all("PCoA_L" not in TOPOS[i] for i in sp["train"])


def test_complexity_and_random():
    sp = make_split(TOPOS, "complexity")
    assert all(len(TOPOS[i]) <= 2 for i in sp["train"]) and all(len(TOPOS[i]) >= 4 for i in sp["test_ood"])
    r = make_split(TOPOS, "random")
    assert len(r["train"]) + len(r["val"]) + len(r["test_id"]) == len(TOPOS)


def test_deterministic():
    a, b = make_split(TOPOS, "variant", seed=2), make_split(TOPOS, "variant", seed=2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
