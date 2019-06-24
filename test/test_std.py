from itertools import product

import pytest
import torch
from torch_scatter import scatter_std

from .utils import grad_dtypes as dtypes, devices, tensor

biases = [True, False]


@pytest.mark.parametrize('dtype,device,bias', product(dtypes, devices, biases))
def test_std(dtype, device, bias):
    src = tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype, device)
    index = tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], torch.long, device)

    out = scatter_std(src, index, dim=-1, unbiased=bias)
    std = src.std(dim=-1, unbiased=bias)[0].item()
    expected = tensor([[std, 0], [0, std]], dtype, device)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_empty_std(dtype, device):
    out = torch.zeros(1, 5, dtype=dtype, device=device)
    src = tensor([], dtype, device).view(0, 5)
    index = tensor([], torch.long, device).view(0, 5)

    out = scatter_std(src, index, dim=0, out=out)
    assert out.tolist() == [[0, 0, 0, 0, 0]]
