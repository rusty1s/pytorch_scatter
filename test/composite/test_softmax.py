from itertools import product

import pytest
import torch
from torch_scatter.composite import scatter_log_softmax, scatter_softmax

from test.utils import devices, tensor, grad_dtypes


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_softmax(dtype, device):
    src = tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_softmax(src, index)

    out0 = torch.softmax(torch.tensor([0.2, 0.2], dtype=dtype), dim=-1)
    out1 = torch.softmax(torch.tensor([0, -2.1, 3.2], dtype=dtype), dim=-1)
    out2 = torch.softmax(torch.tensor([7], dtype=dtype), dim=-1)
    out4 = torch.softmax(torch.tensor([-1, float('-inf')], dtype=dtype),
                         dim=-1)

    expected = torch.stack([
        out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]
    ], dim=0).to(device)

    assert torch.allclose(out, expected)


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_softmax_broadcasting(dtype, device):
    src = torch.randn(10, 5, dtype=dtype, device=device)
    index = tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], torch.long, device)

    out = scatter_softmax(src, index, dim=0).view(5, 2, 5)
    out = out.sum(dim=1)
    assert torch.allclose(out, torch.ones_like(out))


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_log_softmax(dtype, device):
    src = tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_log_softmax(src, index)

    out0 = torch.log_softmax(torch.tensor([0.2, 0.2], dtype=dtype), dim=-1)
    out1 = torch.log_softmax(torch.tensor([0, -2.1, 3.2], dtype=dtype), dim=-1)
    out2 = torch.log_softmax(torch.tensor([7], dtype=dtype), dim=-1)
    out4 = torch.log_softmax(torch.tensor([-1, float('-inf')], dtype=dtype),
                             dim=-1)

    expected = torch.stack([
        out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]
    ], dim=0).to(device)

    assert torch.allclose(out, expected)
