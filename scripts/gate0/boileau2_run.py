"""Boileau 2015 benchmark 2 (common carotid artery) with physics/solver_1d.py vs 3-D reference and six 1D codes.
Usage: python scripts/gate0/boileau2_run.py /path/to/Benchmark2_CommonCarotidArtery
Parameters (paper Tables I, III): L 126 mm, rd 3 mm (Ad = 0.28274 cm^2; our A0 := Ad, pressure offset Pd = 10.933 kPa),
h 0.3 mm, E 700 kPa, rho 1060, mu 4 mPa s, zeta 2 (friction 8 pi mu/rho = scale 1), Windkessel R1 2.4875e8, C 1.7529e-10, R2 1.8697e9
(Pa s/m3, m3/Pa), Pout 0, inflow from inflow.txt (period 1.1 s). Metrics at the midpoint over the last cycle:
relative RMS = sqrt(sum (x - ref)^2 / sum ref^2); also normalised by the reference range (pulse amplitude) because P is offset-dominated."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.solver_1d import build_network, simulate_periodic  # noqa: E402

CODES = ["DCG", "LCG", "FEM", "FVM", "McC", "STM"]


def read_inflow(path, n):
    expr = [l for l in open(path).read().replace("\r", "").split("\n") if l.startswith("inflow=")][0][7:]
    T = 1.1
    t = np.arange(n) * T / n
    return eval(expr, {"__builtins__": {}}, {"sin": np.sin, "cos": np.cos, "PI": math.pi, "t": t, "T": T}) * 1e-6


def run(d, dx=1e-3, scale=1.0):
    n = 1100
    L, rd, h, E, rho, mu = 0.126, 3e-3, 3e-4, 7e5, 1060.0, 4e-3
    pd = 10933.0
    v = dict(name="a", L=L, r0=rd, h=h, E=E)
    net = build_network([v], [], {"a": 1.0}, {"a": dict(R1=2.4875e8, R2=1.8697e9, C=1.7529e-10)}, rho=rho, mu=mu,
                        sigma=0.5, pv=0.0 - pd, wtab=read_inflow(Path(d) / "inflow.txt", n), period=1.1, dx_target=dx)
    net.friction_scale = scale
    out = simulate_periodic(net, 0.0, n_out=n, tol=1e-7, max_cycles=60)
    a = out["a"]
    j = a["A"].shape[1] // 2
    A = a["A"][:, j]
    return dict(P=(pd + a["P"][:, j]) * 1e-3, Q=a["Q"][:, j] * 1e6, dr=(np.sqrt(A / math.pi) - rd) * 1e3,
                diag=out["diagnostics"])


def load(d, code):
    """Last cycle aligned to our samples (phase (i+1) ms, i = 0..1099). 1D codes: files hold the whole run, window
    t in [8.8, 9.9] s (tini, tend from the README), linear interpolation to 1 ms. 3D: one cycle, shifted by the 0.055 s noted in inflow.txt."""
    x = np.loadtxt(Path(d) / "NumData" / f"{code}.txt")
    if code == "3D":
        # inflow.txt says 'add 0.055 to one period to have the minimum flow at t=0': the 3D cycle is shifted by 0.055 s
        # relative to the 1D codes' time axis (sign checked against the six-code mean, which is not our solution)
        x = np.roll(x[:1100], 55 - 1, axis=0)
    else:
        t = 8.8 + (np.arange(1100) + 1) * 1e-3
        x = np.stack([np.interp(t, x[:, 0], x[:, c]) for c in range(x.shape[1])], 1)
    return dict(P=x[:, 1], Q=x[:, 2], dr=x[:, 3])


def rel(a, b, norm="ref"):
    den = np.sqrt(np.sum(b ** 2)) if norm == "ref" else (b.max() - b.min()) * math.sqrt(len(b))
    return 100 * np.sqrt(np.sum((a - b) ** 2)) / den


def main(d):
    ref = {c: load(d, c) for c in ["3D"] + CODES}
    for scale in (1.0, 2.53):
        r = run(d, scale=scale)
        print(f"\nfriction_scale {scale}: converged {r['diag']['converged']} cycles {r['diag']['cycles_to_converge']}")
        for k, name in (("P", "pressure"), ("Q", "flow"), ("dr", "d radius")):
            # ours sample i is at phase (i+1) ms; reference row j at phase j ms
            ours = r[k]
            nref = len(ref["3D"][k])
            ref3 = ref["3D"][k]
            mean6 = np.mean([ref[c][k] for c in CODES], 0)
            s6 = np.mean([rel(ref[c][k], mean6) for c in CODES])
            s6a = np.mean([rel(ref[c][k], mean6, "range") for c in CODES])
            print(f"  {name:9s}: vs 3D {rel(ours, ref3):6.2f}% (range-norm {rel(ours, ref3, 'range'):6.2f}%)  "
                  f"vs six-code mean {rel(ours, mean6):6.2f}% (range-norm {rel(ours, mean6, 'range'):6.2f}%)  "
                  f"six-code spread about mean {s6:5.2f}% (range-norm {s6a:5.2f}%)  [ref rows {nref}]")


if __name__ == "__main__":
    main(sys.argv[1])
