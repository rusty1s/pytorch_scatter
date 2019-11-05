from itertools import product

import torch
import pytest
from torch_scatter import scatter_logsumexp

from .utils import devices, tensor

SUPPORTED_FLOAT_DTYPES = {torch.float32, torch.float64}


@pytest.mark.parametrize('dtype,device', product(SUPPORTED_FLOAT_DTYPES, devices))
def test_logsumexp(dtype, device):
    src = tensor([0.5, 0, 0.5, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_logsumexp(src, index)

    idx0 = torch.logsumexp(torch.tensor([0.5, 0.5], dtype=dtype), dim=-1).tolist()
    idx1 = torch.logsumexp(torch.tensor([0, -2.1, 3.2], dtype=dtype), dim=-1).tolist()
    idx2 = 7   # Single element
    idx3 = torch.finfo(dtype).min   # Empty index, returns yield value
    idx4 = -1   # logsumexp with -inf is the identity

    assert out.tolist() == [idx0, idx1, idx2, idx3, idx4]
