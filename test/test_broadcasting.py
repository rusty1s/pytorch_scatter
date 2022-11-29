from itertools import product

import pytest
import torch
from torch_scatter import scatter
from torch_scatter.testing import devices, reductions


@pytest.mark.parametrize('reduce,device', product(reductions, devices))
def test_broadcasting(reduce, device):
    B, C, H, W = (4, 3, 8, 8)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (B, 1, H, W)).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)
