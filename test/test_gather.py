from itertools import product

import pytest
import torch
from torch_scatter import gather_coo, gather_csr

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward(dtype, device):
    src = tensor([1, 2, 3, 4], dtype, device)
    src = tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype, device)

    indptr = tensor([0, 2, 5, 5, 6], torch.long, device)
    out = gather_csr(src, indptr)
    print('CSR', out)

    index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)
    out = gather_coo(src, index)
    print('COO', out)

    out = src.index_select(0, index)
    print('Expected', out)
