"""Build configs/network_boileau37.yaml (second topology domain) from the Boileau 2015 benchmark-5 supplement.
Input (gitignored third-party data): data/boileau37/mesh.txt, inflow.txt. Output (gitignored): configs/network_boileau37.yaml.
Each of the 37 segments becomes ONE uniform vessel (mean radius, mean pulse-wave speed -> E via c^2 = (2/3) E h/(rho r));
tapering is dropped here (the full tapered network is the verification case in scripts/gate0/boileau37_run.py, not this domain).
Terminals are pure resistors (R1 from mesh column 10 x 1e9, huge R2, tiny C), outlet pressure 3.2 mmHg (Matthys 2007), zeta = 9
friction (the value for this network, see docs/audit.md). Variable 'vessels' are 7 side-branch roots; removing one removes its
whole subtree (unreachable vessels are pruned by physics/cow_network.py). Outflow demand = 1/R1 per terminal.
Usage: python scripts/make_network_boileau37.py"""
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "boileau37"
RHO = 1050.0
VARIABLE = ["s3", "s4", "s9", "s11", "s16", "s24", "s31"]


def main():
    mesh = np.loadtxt(D / "mesh.txt")
    expr = [l for l in open(D / "inflow.txt").read().replace("\r", "").split("\n") if l.startswith("inflow=")][0][7:]
    T = 0.827
    t = np.arange(827) * T / 827
    q = eval(expr, {"__builtins__": {}}, {"sin": np.sin, "cos": np.cos, "PI": math.pi, "t": t, "T": T})  # ml/s
    vessels, nodes, outlets, demand = {}, {}, [], {}
    for sid, na, nb, L, ra, rb, h, ca, cb, r1 in mesh:
        sid = int(sid)
        r, c = 0.5 * (ra + rb), 0.5 * (ca + cb)
        E = 1.5 * RHO * r * c ** 2 / h
        name = f"s{sid}"
        v = dict(length_cm=float(L * 100), r_cm=float(r * 100), h_cm=float(h * 100), E=float(E / 1e6))
        if r1 > 0:
            v.update(R1_Pa_s_m3=float(r1 * 1e9), R2_Pa_s_m3=1e15, C_m3_Pa=1e-3)
            outlets.append(name)
            demand[name] = float(1.0 / (r1 * 1e9) * 1e9)
        vessels[name] = v
        nodes.setdefault(int(na), []).append([name, 0])
        nodes.setdefault(int(nb), []).append([name, 1])
    node_dict = {f"n{k}": ends for k, ends in nodes.items() if len(ends) >= 2}
    cfg = dict(
        constants=dict(rho_kg_m3=RHO, mu_mPa_s=2.5, sigma=0.5, venous_pressure_mmHg=3.2, p_init_mmHg=90, period_s=T,
                       period_range_s=[0.72, 0.95], friction_scale=2.53, dx_target_m=4.0e-3, terminal_capacitor_init="venous", max_cycles=80,
                       source="Boileau et al. 2015 benchmark 5 supplement; Matthys et al. 2007 (mu, p_out)"),
        vessels=vessels,
        inflow=dict(inlet_mean_mL_s={"s1": float(q.mean())}, waveform_table=[float(x) for x in q],
                    note="ascending-aortic flow from inflow.txt (ml/s), unit-mean shape stored, mean in inlet_mean_mL_s"),
        demand=demand,
        topology=dict(nodes=node_dict, inlets=["s1"], outlets=outlets),
        variable_vessels=VARIABLE,
    )
    out = ROOT / "configs" / "network_boileau37.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", out, "vessels", len(vessels), "outlets", len(outlets), "nodes", len(node_dict), "mean inflow", q.mean())


if __name__ == "__main__":
    main()
