"""Gate 0 / topology checks on data/network_topology.py.

Tests marked xfail(strict=True) document KNOWN DEFECTS found in the Gate 0 audit
(docs/audit.md). strict=True means: when the defect is fixed the test XPASSes and the
suite fails until the marker is removed, so a fix cannot go unnoticed.
"""
import pytest

from data.network_topology import NODES, SEGMENTS, junction_map


def _connected(nodes, edges):
    """True if the undirected graph (nodes, edges) is one connected component."""
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(n) for n in nodes}) == 1


def _edges(drop=()):
    return [(s[1], s[2]) for s in SEGMENTS if s[0] not in drop]


def test_segment_names_unique():
    names = [s[0] for s in SEGMENTS]
    assert len(names) == len(set(names))


def test_all_segment_endpoints_are_declared_nodes():
    for name, a, b, *_ in SEGMENTS:
        assert a in NODES and b in NODES, name


def test_complete_network_is_connected():
    assert _connected(NODES, _edges())


def test_junction_map_has_one_entry_per_segment_end():
    jm = junction_map()
    assert sum(len(v) for v in jm.values()) == 2 * len(SEGMENTS)


@pytest.mark.xfail(strict=True, reason="ACoA is a self-loop (start_node == end_node == 'ACoA_mid')")
def test_no_segment_is_a_self_loop():
    loops = [s[0] for s in SEGMENTS if s[1] == s[2]]
    assert loops == []


@pytest.mark.xfail(
    strict=True,
    reason="A1_L and A1_R both end at the shared node 'ACoA_mid', so the left and right "
           "anterior circulation stay connected even when ACoA and both PCoAs are absent",
)
def test_left_right_anterior_disconnected_without_acoa_and_pcoas():
    """Anatomy: without ACoA (and the posterior ring via PCoA) the two carotid systems
    are separate. Graph must reflect that, otherwise 'absent ACoA' is not a topology change."""
    edges = _edges(drop=("ACoA", "PCoA_L", "PCoA_R"))
    nodes = [n for n in NODES]
    # Components containing ICA_L and ICA_R inlets must differ.
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    assert find("ICA_L") != find("ICA_R")
