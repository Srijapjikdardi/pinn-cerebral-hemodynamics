"""Train a K-member surrogate ensemble under a topology split and tabulate per-sample reliability signals and true error.
Usage: python training/reliability_eval.py --data data/cow_v1.h5 --split variant --members 5 --epochs 100 --out outputs/rel_variant.npz
Output npz: err_Q, err_A (per sample, ensemble-mean prediction), signals S1..S6, subset index arrays, truth-signal floors.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.cow_dataset import load  # noqa: E402
from data.splits import make_split  # noqa: E402
from reliability.signals import signals  # noqa: E402
from training.train_surrogate import fit, rel_l2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, Y, M, meta = load(a.data)
    Xt, Yt, Mt = map(torch.as_tensor, (X, Y, M))
    topos = [tuple(sorted(m["absent"])) for m in meta]
    sp = make_split(topos, a.split, seed=0)
    n = len(X)
    s1 = torch.zeros_like(Yt)
    s2 = torch.zeros_like(Yt)
    for k in range(a.members):
        _, best, pred = fit(Xt, Yt, Mt, sp, 100 + k, a.epochs, 256, 4)
        print(f"member {k} best val relL2 {best:.4f}", flush=True)
        s1 += pred
        s2 += pred ** 2
    mean = s1 / a.members
    spread = (s2 / a.members - mean ** 2).clamp_min(0).sqrt()
    ey = rel_l2(mean, Yt, Mt).numpy()
    w = Mt[:, :, None, None, None]
    ysc = (Yt * w).abs().amax(dim=(1, 3, 4), keepdim=True).clamp_min(1e-12)  # per-sample per-vessel-channel scale
    S5 = ((spread * w) / ysc).pow(2).sum(dim=(1, 2, 3, 4)).div((w * torch.ones_like(Yt)).sum(dim=(1, 2, 3, 4))).sqrt().numpy()
    tr = torch.as_tensor(sp["train"])
    xm, xs = Xt[tr].mean(0), Xt[tr].std(0).clamp_min(1e-6)
    Z = (Xt - xm) / xs
    Ztr = Z[tr]
    S6 = np.zeros(n)
    for i in range(0, n, 256):
        d = torch.cdist(Z[i:i + 256], Ztr)
        if i == 0:
            pass
        d[d == 0] = float("inf")  # exclude self for train rows
        S6[i:i + 256] = d.min(1).values.numpy()
    mp = mean.numpy()
    sig = [signals(mp[i], meta[i]) for i in range(n)]
    truth = [signals(Y[i], meta[i]) for i in range(n)]
    out = dict(err_Q=ey[:, 1], err_A=ey[:, 0], S5_ensemble_spread=S5, S6_input_ood=S6,
               n_absent=np.array([len(m["absent"]) for m in meta]),
               physiological=np.array([m["physiological"] for m in meta]))
    for k in sig[0]:
        out[k] = np.array([s[k] for s in sig])
        out["truth_" + k] = np.array([s[k] for s in truth])
    for k, v in sp.items():
        out["idx_" + k] = np.asarray(v)
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
