import torch

from data.cow_dataset import load
from models.graph_surrogate import GraphSurrogate


def test_graph_surrogate_shapes_and_masking(cow_pilot):
    X, Y, M, _ = load(cow_pilot)
    X, M = torch.as_tensor(X), torch.as_tensor(M)
    model = GraphSurrogate(Y.shape[3], Y.shape[4], width=32, rounds=2)
    out = model(X, M).view(len(X), *Y.shape[1:])
    assert out.shape == Y.shape
    assert (out * (1 - M)[:, :, None, None, None]).abs().max() == 0  # absent vessels predict exactly zero


def test_graph_surrogate_ignores_absent_neighbours(cow_pilot):
    """Changing inputs of an absent vessel must not change any prediction (topology enters through the mask)."""
    X, Y, M, _ = load(cow_pilot)
    X, M = torch.as_tensor(X), torch.as_tensor(M)
    i = int((M.sum(1) < M.shape[1]).nonzero()[0]) if (M.sum(1) < M.shape[1]).any() else None
    if i is None:
        return
    model = GraphSurrogate(Y.shape[3], Y.shape[4], width=32, rounds=2).eval()
    v = int((M[i] == 0).nonzero()[0])
    x2 = X[i:i + 1].clone()
    x2[0, 7 + v] += 5.0  # r_scale of the absent vessel (layout: 7 absent flags, then 15 r_scale)
    with torch.no_grad():
        assert torch.allclose(model(X[i:i + 1], M[i:i + 1]), model(x2, M[i:i + 1]), atol=1e-6)
