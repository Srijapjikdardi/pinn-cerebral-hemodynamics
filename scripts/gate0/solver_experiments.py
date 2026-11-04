"""Gate 0 solver experiments on a fixed reference patient (baseline diameters, E=8e5 Pa).

Usage (repo root): python scripts/gate0/solver_experiments.py {timing_demo N | timing_default | topology_effect | mesh NX}
Each experiment prints one JSON line. `mesh NX` also writes mesh_NX.npz in the current directory."""
import sys, time, json, copy, yaml, numpy as np
from pathlib import Path; ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from data.network_topology import SEGMENTS, PatientTopology, default_segment_lengths_mm
from data.generate_synthetic import local_fd_solver

cfg = yaml.safe_load(open(ROOT / "configs" / "demo.yaml"))
def ref_patient(E=8.0e5, absent=()):
    d = {s[0]: (0.0 if s[0] in absent else s[3]) for s in SEGMENTS}
    return PatientTopology(diameters_mm=d, stiffness_E_pa=E, segment_lengths_mm=default_segment_lengths_mm())
def run(patient, n_x=6, n_cycles=1, tpc=50):
    sw = copy.deepcopy(cfg["sweep"]); sw["spatial_points_per_segment"] = n_x
    sw["n_cycles_simulated"] = n_cycles; sw["time_points_per_cycle"] = tpc
    t0 = time.perf_counter(); out = local_fd_solver(patient, sw, cfg["solver"]); return out, time.perf_counter() - t0
def summ(out):
    return {k: dict(Pmax=float(v["P"].max()), Qmax=float(v["Q"].max()), Qmin=float(v["Q"].min()),
                    Qend_absmax=float(np.abs(v["Q"][:, -1]).max())) for k, v in out.items()}
exp = sys.argv[1]
if exp == "timing_demo":
    ts = []
    for i in range(int(sys.argv[2])):
        _, dt = run(ref_patient()); ts.append(dt)
    print(json.dumps(dict(exp=exp, n=len(ts), mean_s=float(np.mean(ts)), std_s=float(np.std(ts)), min_s=min(ts), max_s=max(ts))))
elif exp == "timing_default":
    sw = cfg["sweep"]  # default.yaml resolution: n_x=20, tpc=100, n_cycles=4
    out, dt = run(ref_patient(), n_x=20, n_cycles=4, tpc=100)
    print(json.dumps(dict(exp=exp, n_x=20, n_cycles=4, wall_s=dt)))
elif exp == "topology_effect":
    base, _ = run(ref_patient())
    for absent in (("ACoA",), ("PCoA_L",), ("PCoA_L", "PCoA_R")):
        alt, _ = run(ref_patient(absent=absent))
        res = {}
        for k in alt:
            if k in base:
                n = min(base[k]["P"].shape[0], alt[k]["P"].shape[0])
                num = np.abs(base[k]["P"][:n] - alt[k]["P"][:n]).max(); den = np.abs(base[k]["P"][:n]).max()
                qn = np.abs(base[k]["Q"][:n] - alt[k]["Q"][:n]).max(); qd = np.abs(base[k]["Q"][:n]).max()
                res[k] = dict(relP=float(num/den), relQ=float(qn/qd))
        print(json.dumps(dict(exp=exp, absent=list(absent), max_relP=max(r["relP"] for r in res.values()),
                              max_relQ=max(r["relQ"] for r in res.values()), per_seg=res)))
elif exp == "mesh":
    n_x = int(sys.argv[2]); out, dt = run(ref_patient(), n_x=n_x)
    np.savez(f"mesh_{n_x}.npz",
             **{f"{k}_{f}": v[f] for k, v in out.items() for f in ("t", "x", "P", "Q", "A")})
    print(json.dumps(dict(exp=exp, n_x=n_x, wall_s=dt, summary=summ(out))))
