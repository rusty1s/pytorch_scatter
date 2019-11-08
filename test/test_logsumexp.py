from itertools import product

import torch
import pytest
from torch_scatter import scatter_logsumexp

from .utils import devices, tensor, grad_dtypes


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_logsumexp(dtype, device):
    src = tensor([0.5, 0, 0.5, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_logsumexp(src, index)

    out0 = torch.logsumexp(torch.tensor([0.5, 0.5], dtype=dtype), dim=-1)
    out1 = torch.logsumexp(torch.tensor([0, -2.1, 3.2], dtype=dtype), dim=-1)
    out2 = torch.logsumexp(torch.tensor(7, dtype=dtype), dim=-1)
    out3 = torch.tensor(torch.finfo(dtype).min, dtype=dtype)
    out4 = torch.tensor(-1, dtype=dtype)

    expected = torch.stack([out0, out1, out2, out3, out4], dim=0)
    assert torch.allclose(out, expected)
