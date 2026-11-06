"""Data-only MLP surrogate on the verified CoW dataset, trained under a chosen topology split.
Usage: python training/train_surrogate.py --data data/cow_v1.h5 --split variant --seeds 3 --epochs 300 --out outputs/sur_variant.json
Metric: per-sample relative L2 of Q and of A (over present vessels), reported per subset (val, test_id, test_ood).
Model selection on val only (val never contains held-out topologies). No physics loss: this is the baseline the
reliability checker will be evaluated against.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.cow_dataset import load  # noqa: E402
from data.splits import make_split  # noqa: E402


class MLP(nn.Module):
    def __init__(self, din, dout, width=256, depth=4):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def rel_l2(pred, y, m):
    """pred,y: (N,V,2,NT,NX); m: (N,V). Per-sample, per-channel relative L2 over present vessels."""
    w = m[:, :, None, None, None]
    num = (((pred - y) * w) ** 2).sum(dim=(1, 3, 4)).sqrt()
    den = ((y * w) ** 2).sum(dim=(1, 3, 4)).sqrt().clamp_min(1e-12)
    return num / den  # (N,2)


def fit(X, Y, M, sp, seed, epochs, width, depth, lr=2e-3, bs=64, phys=None, phys_w=0.0, project=False):
    torch.manual_seed(seed)
    tr = torch.as_tensor(sp["train"])
    xm, xs = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    ys = Y[tr].flatten(0, 0).abs().amax(dim=(0, 3, 4), keepdim=True).clamp_min(1e-12)  # (1,V,2,1,1) scale
    Xn, Yn = (X - xm) / xs, Y / ys
    flat = Yn.shape[1:].numel()
    model = MLP(X.shape[1], flat, width, depth)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    va = torch.as_tensor(sp["val"])

    def predict(idx):
        with torch.no_grad():
            out = model(Xn[idx]).view(len(idx), *Yn.shape[1:])
            if project:
                from training.physics_loss import project_junctions
                out = project_junctions(out, ys, idx, M, phys)
            return out

    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            out = model(Xn[b]).view(len(b), *Yn.shape[1:])
            if project:
                from training.physics_loss import project_junctions
                out = project_junctions(out, ys, b, M, phys)
            loss = (((out - Yn[b]) * M[b][:, :, None, None, None]) ** 2).mean()
            if phys is not None and phys_w > 0:
                from training.physics_loss import loss as plos
                lm, lj = plos(out, ys, b, M, phys)
                loss = loss + phys_w * (lm + lj)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            v = rel_l2(predict(va) * ys, Y[va], M[va]).mean().item()
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()

    def evaluate(idx):
        idx = torch.as_tensor(idx)
        if len(idx) == 0:
            return None
        e = rel_l2(predict(idx) * ys, Y[idx], M[idx])
        return dict(n=len(idx), A_mean=e[:, 0].mean().item(), Q_mean=e[:, 1].mean().item(),
                    Q_p90=e[:, 1].quantile(0.9).item(), Q_median=e[:, 1].median().item())
    pred_all = torch.cat([predict(torch.arange(i, min(i + 128, len(X)))) * ys for i in range(0, len(X), 128)])
    return {k: evaluate(sp[k]) for k in ("val", "test_id", "test_ood")}, best, pred_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="variant")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--physiological-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, Y, M, meta = load(a.data, a.physiological_only)
    X, Y, M = map(torch.as_tensor, (X, Y, M))
    topos = [tuple(sorted(m["absent"])) for m in meta]
    sp = make_split(topos, a.split, seed=0)
    print({k: len(v) for k, v in sp.items()}, flush=True)
    res = []
    for s in range(a.seeds):
        r, best, _ = fit(X, Y, M, sp, s, a.epochs, a.width, a.depth)
        res.append(r)
        print(s, json.dumps(r), flush=True)
    summ = {k: {m: float(np.mean([r[k][m] for r in res])) for m in ("A_mean", "Q_mean", "Q_median", "Q_p90")}
            for k in ("val", "test_id", "test_ood") if res[0][k]}
    print("mean over seeds:", json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(a), seeds=res, summary=summ), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
