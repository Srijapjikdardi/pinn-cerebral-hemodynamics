"""Small shared helpers for building conditioning vectors / junction batches
outside the main training loop (e.g. for notebooks or quick inference scripts)."""

import numpy as np
from data.network_topology import SEGMENTS, junction_map


def example_conditioning_vector(missing_segments=None) -> np.ndarray:
    """Convenience constructor for ad-hoc experimentation, e.g. in a notebook:

        cond = example_conditioning_vector(missing_segments=["PCoA_L", "ACoA"])
    """
    from data.network_topology import PatientTopology
    missing = set(missing_segments or [])
    diameters = {s[0]: (0.0 if s[0] in missing else s[3]) for s in SEGMENTS}
    return PatientTopology(diameters_mm=diameters, stiffness_E_pa=8.0e5).conditioning_vector()


def describe_junctions() -> str:
    """Human-readable printout of every junction and the segments meeting there —
    useful for sanity-checking models/losses.py:junction_loss wiring."""
    lines = []
    for node, touching in junction_map().items():
        segs = [f"{name}({'in' if sign == 1 else 'out'})" for name, sign in touching]
        if len(segs) >= 2:
            lines.append(f"{node}: " + ", ".join(segs))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_junctions())
