"""
Phase 4.1 / 4.2 — Validation & speed benchmarking.

4.1: Run the trained G-PINN on the held-out synthetic patients (the last
     `sweep.n_validation_holdout` patients in the dataset, by convention —
     make sure generate_synthetic.py's sweep and this script agree on the split)
     and report Mean Absolute Percentage Error (MAPE) on peak systolic pressure.

4.2: Benchmark wall-clock time for a forward pass of the G-PINN vs. the time
     the reference solver (local_fd_solver or SimVascular) took to produce the
     same patient's simulation. This is what substantiates a "100x speedup"
     style claim in the paper — report it honestly, with hardware specs.
"""

import argparse
import time
import yaml
import h5py
import numpy as np
import torch

from data.network_topology import SEGMENTS
from models.gpinn import build_model

SEGMENT_NAMES = [s[0] for s in SEGMENTS]
SEG_ID_LOOKUP = {name: i for i, name in enumerate(SEGMENT_NAMES)}


def peak_systolic_pressure(P_series: np.ndarray) -> float:
    """P_series: (n_time, n_x) array for one segment across one cardiac cycle.
    Returns the peak pressure anywhere along the segment during the cycle."""
    return float(np.max(P_series))


def compute_mape(model, patients, device) -> dict:
    errors = []
    for p in patients:
        cond = torch.tensor(p["cond"], dtype=torch.float32, device=device).unsqueeze(0)
        for seg_name, seg in p["segments"].items():
            true_peak = peak_systolic_pressure(seg["P"])
            n_t, n_x = seg["P"].shape
            x = torch.linspace(0, 1, n_x, device=device).unsqueeze(1)
            seg_id = torch.full((n_x,), SEG_ID_LOOKUP[seg_name], dtype=torch.long, device=device)
            cond_rep = cond.repeat(n_x, 1)

            peaks = []
            for it in range(n_t):
                t_val = torch.full((n_x, 1), it / max(n_t - 1, 1), device=device)
                with torch.no_grad():
                    out = model(x, t_val, seg_id, cond_rep)
                peaks.append(out["P"].max().item())
            pred_peak = max(peaks)

            if abs(true_peak) > 1e-8:
                errors.append(abs(pred_peak - true_peak) / abs(true_peak) * 100.0)

    return {
        "mape_percent": float(np.mean(errors)) if errors else float("nan"),
        "n_samples": len(errors),
    }


def benchmark_speed(model, patients, device, n_trials=5, solver_times_s=None) -> dict:
    """Compares G-PINN full-patient forward-pass time vs. the solver's recorded
    per-patient wall-clock time (`solver_time_s`, stored by generate_synthetic.py
    for every patient — this WAS being ignored here despite already existing in the
    dataset; fixed to actually read and use it).

    "Full-patient" means: every present segment, at every timepoint the solver
    actually computed for that patient — not one segment at one timestep — since
    that's the unit being replaced for a real speedup claim."""

    def full_patient_forward(patient):
        for seg_name, seg in patient["segments"].items():
            n_t, n_x = seg["A"].shape
            x = torch.linspace(0, 1, n_x, device=device).unsqueeze(1)
            seg_id = torch.full((n_x,), SEG_ID_LOOKUP[seg_name], dtype=torch.long, device=device)
            cond = torch.tensor(patient["cond"], dtype=torch.float32, device=device).unsqueeze(0).repeat(n_x, 1)
            for it in range(n_t):
                t_val = torch.full((n_x, 1), it / max(n_t - 1, 1), device=device)
                model(x, t_val, seg_id, cond)

    # warmup
    with torch.no_grad():
        full_patient_forward(patients[0])

    per_patient_times = []
    with torch.no_grad():
        for p in patients[:n_trials]:
            start = time.perf_counter()
            full_patient_forward(p)
            per_patient_times.append(time.perf_counter() - start)

    gpinn_mean_s = float(np.mean(per_patient_times))
    result = {
        "gpinn_full_patient_forward_pass_s": gpinn_mean_s,
        "n_patients_timed": len(per_patient_times),
    }

    if solver_times_s:
        solver_mean_s = float(np.mean(solver_times_s))
        result["solver_mean_s"] = solver_mean_s
        result["solver_n_patients"] = len(solver_times_s)
        result["speedup_x"] = solver_mean_s / gpinn_mean_s if gpinn_mean_s > 0 else float("inf")
    else:
        result["note"] = ("No solver_time_s found in this dataset (older generation run, "
                           "or generate_synthetic.py predates that fix) — cannot compute a "
                           "real speedup ratio, only the raw G-PINN forward-pass time.")

    return result


def evaluate(config_path, checkpoint_path, data_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg["topology"], cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    patients = []
    solver_times_s = []
    with h5py.File(data_path, "r") as h5:
        keys = sorted(h5.keys())
        holdout_keys = keys[-cfg["sweep"]["n_validation_holdout"]:]
        for key in holdout_keys:
            grp = h5[key]
            cond = grp["conditioning"]["vector"][:]
            segs = {sk[len("segment_"):]: {k: grp[sk][k][:] for k in grp[sk].keys()}
                    for sk in grp.keys() if sk.startswith("segment_")}
            patients.append({"cond": cond, "segments": segs})
            if "solver_time_s" in grp.attrs:
                solver_times_s.append(float(grp.attrs["solver_time_s"]))

    mape_results = compute_mape(model, patients, device)
    speed_results = benchmark_speed(model, patients, device, solver_times_s=solver_times_s)

    print("=== Phase 4.1: MAPE on held-out synthetic patients ===")
    print(mape_results)
    target = cfg["validation"]["mape_target_percent"]
    verdict = "PASS" if mape_results["mape_percent"] < target else "FAIL"
    print(f"Target: < {target}% — {verdict}")

    print("\n=== Phase 4.2: Speed benchmark ===")
    print(speed_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/gpinn_final.pt")
    parser.add_argument("--data", default="data/synthetic_dataset.h5")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.data)
