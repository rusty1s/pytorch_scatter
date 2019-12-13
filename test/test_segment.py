from itertools import product

import pytest
import torch
from torch_scatter import segment_add

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward(dtype, device):
    src = tensor([1, 2, 3, 4, 5, 6], dtype, device)
    index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)

    out, key = segment_add(src, index, dim=0)
    print(out)
    print(key)
