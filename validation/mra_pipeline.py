"""
Phase 4.3 — Real-world MRA pipeline (the "final flourish").

This is intentionally a thin script, not a full segmentation pipeline: the plan
calls for *manual* centerline extraction in 3D Slicer for 1-2 patients, not an
automated segmentation model. Workflow:

  1. Load the patient's MRA (DICOM/NIfTI) into 3D Slicer.
  2. Use the "Vessel Centerline Extraction" or "Extract Centerline" module
     (part of the SlicerVMTK extension) to trace each of the 11 CoW segments.
  3. For each traced segment, record:
       - length (mm) — sum of centerline segment lengths
       - diameter (mm) — average cross-sectional diameter along the centerline
       - presence (bool) — whether Slicer could trace a continuous centerline at all
  4. Save those measurements into a JSON file matching PATIENT_MEASUREMENTS_SCHEMA
     below (one file per patient).
  5. Run this script to convert that JSON into a PatientTopology and get a
     G-PINN pressure/flow prediction.

This script does NOT talk to 3D Slicer programmatically — it consumes the JSON
you export by hand (or via a Slicer Python console script, which is a natural
follow-up automation once you've done this manually for a couple of patients).
"""

import argparse
import json
import yaml
import numpy as np
import torch

from data.network_topology import SEGMENTS, PatientTopology
from models.gpinn import build_model

SEGMENT_NAMES = [s[0] for s in SEGMENTS]
SEG_ID_LOOKUP = {name: i for i, name in enumerate(SEGMENT_NAMES)}

PATIENT_MEASUREMENTS_SCHEMA = {
    "patient_id": "string",
    "estimated_wall_stiffness_E_pa": "float — see note below on estimating this from age",
    "segments": {
        "<segment_name e.g. 'ICA_L'>": {
            "diameter_mm": "float, 0.0 if segment could not be traced (absent/hypoplastic)",
            "length_mm": "float",
        }
    },
}


def estimate_stiffness_from_age(age_years: float) -> float:
    """Very rough linear mapping from patient age to arterial stiffness E (Pa),
    calibrated to the sweep range in configs/default.yaml (4e5 Pa at ~20yo up to
    2e6 Pa at ~90yo). This is a placeholder — replace with a validated
    age-stiffness relationship (e.g. from pulse-wave-velocity literature) before
    using this clinically."""
    age = np.clip(age_years, 20, 90)
    return float(4.0e5 + (age - 20) / (90 - 20) * (2.0e6 - 4.0e5))


def load_patient_json(json_path: str) -> PatientTopology:
    with open(json_path) as f:
        data = json.load(f)

    diameters, lengths = {}, {}
    for seg_name in SEGMENT_NAMES:
        seg_data = data["segments"].get(seg_name, {"diameter_mm": 0.0, "length_mm": 10.0})
        diameters[seg_name] = seg_data["diameter_mm"]
        lengths[seg_name] = seg_data["length_mm"]

    E = data.get("estimated_wall_stiffness_E_pa") or estimate_stiffness_from_age(data.get("age_years", 60))
    return PatientTopology(diameters_mm=diameters, stiffness_E_pa=E, segment_lengths_mm=lengths)


def predict(config_path, checkpoint_path, patient_json_path, n_time_points=100):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg["topology"], cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    patient = load_patient_json(patient_json_path)
    cond = torch.tensor(patient.conditioning_vector(), dtype=torch.float32, device=device).unsqueeze(0)

    results = {}
    for seg_name, d_mm in patient.diameters_mm.items():
        if d_mm <= 1e-6:
            continue  # segment absent for this patient — no prediction to make
        x = torch.linspace(0, 1, 20, device=device).unsqueeze(1)
        seg_id = torch.full((20,), SEG_ID_LOOKUP[seg_name], dtype=torch.long, device=device)
        cond_rep = cond.repeat(20, 1)

        peak_pressures = []
        for it in range(n_time_points):
            t_val = torch.full((20, 1), it / (n_time_points - 1), device=device)
            with torch.no_grad():
                out = model(x, t_val, seg_id, cond_rep)
            peak_pressures.append(out["P"].max().item())
        results[seg_name] = {
            "peak_systolic_pressure_pa": max(peak_pressures),
            "min_diastolic_pressure_pa": min(peak_pressures),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/gpinn_final.pt")
    parser.add_argument("--patient_json", required=True,
                         help="Path to a JSON file matching PATIENT_MEASUREMENTS_SCHEMA, "
                              "hand-exported from 3D Slicer centerline extraction.")
    args = parser.parse_args()
    result = predict(args.config, args.checkpoint, args.patient_json)
    print(json.dumps(result, indent=2))
