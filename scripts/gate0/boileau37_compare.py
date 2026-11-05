"""Compare outputs/boileau37_*.npz with the paper's in vitro data (ExpData) and six numerical schemes (NumData).
Usage: python scripts/gate0/boileau37_compare.py outputs/boileau37_pout0.npz /path/to/Benchmark5_37ArteryNetwork
Site -> segment mapping is not in the supplement (it is in the paper's Table IX / Fig. 5, not available here); it is inferred by
matching our midpoint waveform to the six-scheme MEAN (not to the in vitro data), and reported."""
import codecs
import re
import sys

import numpy as np

SITES = ["AorticArchII", "ThoracicAortaII", "LSubclavianI", "RIliacFemoralII", "LUlnar", "RAnteriorTibial", "RUlnar", "Splenic"]
SCH = ["DCG", "LCG", "FEM", "FVM", "McC", "STM"]
T = 0.827


def shifts(root):
    txt = codecs.open(f"{root}/time_shifts.txt", "r", "utf-16").read()
    out = {}
    for m in re.finditer(r"(\w+)\s*\n\s*tini\s*=\s*([\d.]+)\s*s\s*\n\s*tend\s*=\s*([\d.]+)", txt):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def sim_at(arr, tabs):
    ph = (tabs % T) / T * 827
    x = np.arange(827) + 1.0
    return np.interp(ph, np.concatenate([x - 827, x, x + 827]), np.tile(arr, 3))


def rel(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)), np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())


def main(npz, root):
    d = np.load(npz)
    sh = shifts(root)
    num = {}
    for s in SCH:
        for y in "PQ":
            num[(s, y)] = np.loadtxt(f"{root}/NumData/{s}_{y}.txt")
    print(f"{npz}: periodic change {float(d['change']):.1e}, vol in {float(d['vol_in'])*1e6:.2f} ml out {float(d['vol_out'])*1e6:.2f} ml (whole run)")
    best = {}
    for k, site in enumerate(SITES):
        t0, t1 = sh[site]
        tref = np.arange(t0, t1, 0.001)
        cons = {y: np.mean([np.interp(tref, num[(s, y)][:, 0], num[(s, y)][:, 1 + k]) for s in SCH], 0) for y in "PQ"}
        errs = []
        for seg in range(37):
            e = sum(rel(sim_at(d[y][:, seg], tref), cons[y])[0] for y in "PQ")
            errs.append(e)
        best[site] = int(np.argmin(errs)) + 1
        srt = np.sort(errs)
        print(f"{site:16s} best segment {best[site]:2d} (combined rel-RMS {srt[0]:.3f}; next {srt[1]:.3f})")
    print("\nrelative RMS (/RMS of reference, /range of reference) at inferred segment midpoint, last-cycle window")
    print(f"{'site':16s} {'seg':>3s} | {'vs in vitro P':>16s} {'vs in vitro Q':>16s} | {'vs 6-scheme mean P':>19s} {'Q':>14s} | {'scheme spread P':>15s} {'Q':>8s}")
    for k, site in enumerate(SITES):
        t0, t1 = sh[site]
        seg = best[site] - 1
        row = []
        ex = {}
        for y in "PQ":
            e = np.loadtxt(f"{root}/ExpData/{site}_{y}.txt")
            ex[y] = (e[:, 0] + t0, e[:, 1])
        tref = np.arange(t0, t1, 0.001)
        cons, spread = {}, {}
        for y in "PQ":
            allv = np.array([np.interp(tref, num[(s, y)][:, 0], num[(s, y)][:, 1 + k]) for s in SCH])
            cons[y] = allv.mean(0)
            spread[y] = np.mean([rel(v, cons[y])[0] for v in allv])
        pe = rel(sim_at(d["P"][:, seg], ex["P"][0]), ex["P"][1])
        qe = rel(sim_at(d["Q"][:, seg], ex["Q"][0]), ex["Q"][1])
        pc = rel(sim_at(d["P"][:, seg], tref), cons["P"])
        qc = rel(sim_at(d["Q"][:, seg], tref), cons["Q"])
        print(f"{site:16s} {best[site]:3d} | {pe[0]*100:6.1f}% /{pe[1]*100:5.1f}% {qe[0]*100:6.1f}% /{qe[1]*100:5.1f}% | "
              f"{pc[0]*100:8.1f}% /{pc[1]*100:5.1f}% {qc[0]*100:6.1f}% | {spread['P']*100:14.1f}% {spread['Q']*100:7.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
