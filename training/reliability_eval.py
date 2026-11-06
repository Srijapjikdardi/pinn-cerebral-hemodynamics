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
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--phys-weight", type=float, default=0.0)
    ap.add_argument("--project", action="store_true", help="hard junction-mass projection layer")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "graph"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4, help="MLP hidden layers / graph message-passing rounds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, Y, M, meta = load(a.data)
    Xt, Yt, Mt = map(torch.as_tensor, (X, Y, M))
    topos = [tuple(sorted(m["absent"])) for m in meta]
    sp = make_split(topos, a.split, seed=0)
    n = len(X)
    phys = None
    if a.phys_weight > 0 or a.project:
        from training import physics_loss
        phys = physics_loss.build(meta, Y.shape[3], Y.shape[4])
    s1 = torch.zeros_like(Yt)
    s2 = torch.zeros_like(Yt)
    for k in range(a.members):
        r = fit(Xt, Yt, Mt, sp, a.seed_base + k, a.epochs, a.width, a.depth, lr=a.lr, phys=phys, phys_w=a.phys_weight, project=a.project, arch=a.arch)
        best, pred = r[1], r[2]
        if a.project:
            raw_sum = r[3] if k == 0 else raw_sum + r[3]
        print(f"member {k} best val relL2 {best:.4f}", flush=True)
        s1 += pred
        s2.addcmul_(pred, pred)
        del pred, r
    mean = s1.div_(a.members)  # in place: keeps peak memory near 4 field copies (OOM on 8 GB otherwise)
    spread = s2.div_(a.members).sub_(mean ** 2).clamp_min_(0).sqrt_()
    ey = rel_l2(mean, Yt, Mt).numpy()
    S5 = np.zeros(n)
    nfield = Yt.shape[2] * Yt.shape[3] * Yt.shape[4]
    for i in range(0, n, 128):  # per-sample per-vessel-channel scale; RMS of normalised spread over present vessels
        sl = slice(i, i + 128)
        w = Mt[sl, :, None, None, None]
        ysc = (Yt[sl] * w).abs().amax(dim=(1, 3, 4), keepdim=True).clamp_min(1e-12)
        S5[sl] = ((spread[sl] * w) / ysc).pow(2).sum(dim=(1, 2, 3, 4)).div(w.sum(dim=(1, 2, 3, 4)) * nfield).sqrt().numpy()
    del spread
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
    if a.project:  # mean over members of the raw-output junction violation (before the hard-constraint layer)
        out["S8_projection_correction"] = (raw_sum / a.members).numpy()
    for k, v in sp.items():
        out["idx_" + k] = np.asarray(v)
    keep = np.concatenate([sp['val'], sp['test_id'], sp['test_ood']]).astype(int)
    out['pred_idx'] = keep
    np.savez(a.out, **out)
    # ensemble-mean predictions (val/test) for later signal computation; large (~80 MB), gitignored
    np.savez_compressed(a.out.replace(".npz", "_pred.npz"), pred_idx=keep, pred_mean=mp[keep].astype(np.float16))
    print("saved", a.out)


if __name__ == "__main__":
    main()
