"""
Circle of Willis (CoW) 1D network topology.

Defines the fixed graph structure (which segments exist in a *complete* circle,
how they connect at junction nodes) and the machinery for generating
patient-specific *variants* of that graph by zeroing out optional segments
(anatomical variations, e.g. missing PCoA — present in ~50% of the population).

This is intentionally NOT a general-purpose graph library: the CoW topology is
fixed and small (11 segments, 8 named nodes), so we hardcode it rather than
parse it from a generic format.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# ------------------------------------------------------------------
# Node names (junction points where vessels meet)
# ------------------------------------------------------------------
NODES = [
    "ICA_L", "ICA_R",        # internal carotid inlets
    "BA",                     # basilar artery inlet (vertebrobasilar)
    "ACA_A1_L_jn", "ACA_A1_R_jn",
    "MCA_L_jn", "MCA_R_jn",
    "PCA_P1_L_jn", "PCA_P1_R_jn",
    "ACoA_mid",
]

# ------------------------------------------------------------------
# Segment (edge) definitions: (name, start_node, end_node, baseline_diameter_mm, is_optional)
# Baseline diameters are adult population averages (Krabbe-Hartkamp et al., 1998 —
# used here only as a *plausible default*, not a literature guarantee; adjust
# for your cohort).
# ------------------------------------------------------------------
SEGMENTS: List[Tuple[str, str, str, float, bool]] = [
    ("ICA_L",     "ICA_L",       "ACA_A1_L_jn", 4.5, False),
    ("ICA_R",     "ICA_R",       "ACA_A1_R_jn", 4.5, False),
    ("ACA_A1_L",  "ACA_A1_L_jn", "ACoA_mid",    2.2, False),
    ("ACA_A1_R",  "ACA_A1_R_jn", "ACoA_mid",    2.2, False),
    ("ACoA",      "ACoA_mid",    "ACoA_mid",    1.5, True),   # anterior communicating (often hypoplastic/absent)
    ("MCA_L",     "ACA_A1_L_jn", "MCA_L_jn",    3.0, False),
    ("MCA_R",     "ACA_A1_R_jn", "MCA_R_jn",    3.0, False),
    ("PCoA_L",    "ACA_A1_L_jn", "PCA_P1_L_jn", 1.4, True),   # posterior communicating (~50% pop. hypoplastic)
    ("PCoA_R",    "ACA_A1_R_jn", "PCA_P1_R_jn", 1.4, True),
    ("PCA_P1_L",  "BA",          "PCA_P1_L_jn", 2.1, False),
    ("PCA_P1_R",  "BA",          "PCA_P1_R_jn", 2.1, False),
]

OPTIONAL_SEGMENTS = [s[0] for s in SEGMENTS if s[4]]


@dataclass
class PatientTopology:
    """A single patient's instantiated CoW graph."""
    diameters_mm: Dict[str, float] = field(default_factory=dict)   # segment_name -> diameter (0 = absent)
    stiffness_E_pa: float = 8.0e5
    segment_lengths_mm: Dict[str, float] = field(default_factory=dict)

    def adjacency_vector(self) -> np.ndarray:
        """Flattened [diameter_1..diameter_11] conditioning vector, 0 = vessel absent."""
        return np.array([self.diameters_mm[s[0]] for s in SEGMENTS], dtype=np.float32)

    def presence_mask(self) -> np.ndarray:
        return (self.adjacency_vector() > 1e-6).astype(np.float32)

    def conditioning_vector(self) -> np.ndarray:
        """Vector fed to the G-PINN's graph encoder: diameters + presence mask + stiffness."""
        return np.concatenate([
            self.adjacency_vector() / 5.0,       # normalize by ~max diameter (mm)
            self.presence_mask(),
            np.array([self.stiffness_E_pa / 2.0e6], dtype=np.float32),  # normalize by sweep max
        ])


def default_segment_lengths_mm() -> Dict[str, float]:
    """Rough adult-average segment lengths (mm). Replace with per-patient MRA measurements
    when running the real-patient pipeline (Phase 4.3)."""
    return {
        "ICA_L": 15.0, "ICA_R": 15.0,
        "ACA_A1_L": 12.0, "ACA_A1_R": 12.0,
        "ACoA": 2.5,
        "MCA_L": 16.0, "MCA_R": 16.0,
        "PCoA_L": 15.0, "PCoA_R": 15.0,
        "PCA_P1_L": 7.0, "PCA_P1_R": 7.0,
    }


# Fixed per-segment length lookup, in meters, in SEGMENTS order — used to correct
# PDE-residual derivatives for the fact that collocation x is sampled normalized to
# [0,1] rather than physical meters (see models/losses.py:pde_residuals). Same length
# table for every patient (default_segment_lengths_mm(), not per-patient MRA
# measurements), consistent with what data/generate_synthetic.py actually used to
# build the Phase 1 ground truth.
SEGMENT_LENGTHS_M: List[float] = [
    default_segment_lengths_mm()[s[0]] / 1000.0 for s in SEGMENTS
]


def make_random_patient(rng: np.random.Generator, sweep_cfg: dict) -> PatientTopology:
    """Sample one synthetic patient: randomized stiffness + random optional-vessel absence,
    per the ranges in configs/default.yaml -> sweep."""
    diameters = {}
    for name, _, _, baseline_mm, is_optional in SEGMENTS:
        if is_optional and rng.random() < sweep_cfg["missing_vessel_prob"]:
            diameters[name] = 0.0
        else:
            # +/-20% physiological variation around baseline
            diameters[name] = float(baseline_mm * rng.uniform(0.8, 1.2))

    E = float(rng.uniform(*sweep_cfg["stiffness_E_range_Pa"]))
    return PatientTopology(
        diameters_mm=diameters,
        stiffness_E_pa=E,
        segment_lengths_mm=default_segment_lengths_mm(),
    )


def junction_map() -> Dict[str, List[Tuple[str, int]]]:
    """
    For each node, list the (segment_name, sign) pairs touching it, where sign=+1 if the
    segment's flow convention points *into* the node and -1 if it points *out*.
    Used to build Kirchhoff (mass-conservation) constraints at each junction — see
    models/losses.py:junction_loss.
    """
    jmap: Dict[str, List[Tuple[str, int]]] = {n: [] for n in NODES}
    for name, start, end, _, _ in SEGMENTS:
        jmap[start].append((name, -1))  # flow leaves 'start'
        jmap[end].append((name, +1))    # flow arrives at 'end'
    return jmap


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    cfg = {"missing_vessel_prob": 0.35, "stiffness_E_range_Pa": [4.0e5, 2.0e6]}
    p = make_random_patient(rng, cfg)
    print("Sample patient diameters (mm):", p.diameters_mm)
    print("Conditioning vector shape:", p.conditioning_vector().shape)
    print("Junction map:", junction_map())
