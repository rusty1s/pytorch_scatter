import torch
from torch_scatter import scatter_max, scatter_min


def test_max_fill_value():
    src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

    out, _ = scatter_max(src, index)

    v = torch.finfo(torch.float).min
    assert out.tolist() == [[v, v, 4, 3, 2, 0], [2, 4, 3, v, v, v]]


def test_min_fill_value():
    src = torch.Tensor([[-2, 0, -1, -4, -3], [0, -2, -1, -3, -4]])
    index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

    out, _ = scatter_min(src, index)

    v = torch.finfo(torch.float).max
    assert out.tolist() == [[v, v, -4, -3, -2, 0], [-2, -4, -3, v, v, v]]
