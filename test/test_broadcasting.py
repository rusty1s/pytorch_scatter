import pytest
import torch
from torch_scatter import scatter_add

from .utils import devices


@pytest.mark.parametrize('device', devices)
def test_broadcasting(device):
    B, C, H, W = (4, 3, 8, 8)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (B, 1, H, W)).to(device, torch.long)
    out = scatter_add(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, 1, H, W), device=device)
    index = torch.randint(0, H, (B, C, H, W)).to(device, torch.long)
    out = scatter_add(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, 1, H, W), device=device)
    index = torch.randint(0, H, (B, 1, H, W)).to(device, torch.long)
    out = scatter_add(src, index, dim=2, dim_size=H)
    assert out.size() == (B, 1, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = scatter_add(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)
