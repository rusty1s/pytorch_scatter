import pytest
import torch
from torch_scatter import scatter_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='No multiple GPUS')
def test_multi_gpu():
    device = torch.device('cuda:1')
    src = torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)
    assert scatter_max(src, index)[0].tolist() == [3, 5]
