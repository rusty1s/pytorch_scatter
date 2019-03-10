from itertools import product

import pytest
import torch
from torch_scatter import scatter_std

from .utils import grad_dtypes as dtypes, devices, tensor

biass = [True, False]


@pytest.mark.parametrize('dtype,device,bias', product(dtypes, devices, biass))
def test_std(dtype, device, bias):
    src = tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype, device)
    index = tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], torch.long, device)

    out = scatter_std(src, index, dim=-1, unbiased=bias)
    std = src.std(dim=-1, unbiased=bias)[0].item()
    expected = torch.tensor([[std, 0], [0, std]], dtype=out.dtype)
    assert torch.allclose(out, expected)
