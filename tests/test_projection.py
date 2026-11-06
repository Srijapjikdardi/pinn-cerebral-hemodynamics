import torch

from data.cow_dataset import load
from training import physics_loss as pl


def test_projection_zeroes_junction_residual_and_keeps_true_fields():
    X, Y, M, meta = load("data/cow_pilot.h5")
    Yt, Mt = torch.as_tensor(Y), torch.as_tensor(M)
    P = pl.build(meta, Y.shape[3], Y.shape[4])
    ys = Yt.abs().amax(dim=(0, 3, 4), keepdim=True).clamp_min(1e-12)
    idx = torch.arange(len(Y))
    noisy = Yt / ys * (1 + 0.1 * torch.randn_like(Yt))
    proj = pl.project_junctions(noisy, ys, idx, Mt, P)
    _, j_before = pl.loss(noisy, ys, idx, Mt, P)
    _, j_after = pl.loss(proj, ys, idx, Mt, P)
    assert j_before > 1e-5 and j_after < 1e-12 * max(1.0, float(j_before))
    # A channel and absent vessels untouched
    assert torch.allclose(proj[:, :, 0], noisy[:, :, 0], atol=1e-6)
    assert (proj * (1 - Mt)[:, :, None, None, None]).abs().max() < 1e-6  # absent vessels stay zero
    # exact data: the projection should change almost nothing (true end flows nearly conserve at cell centres)
    pt = pl.project_junctions(Yt / ys, ys, idx, Mt, P)
    assert (pt - Yt / ys).abs().max() < 0.05
