from itertools import product

import pytest
import torch
from torch_scatter import segment_coo, segment_csr

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward(dtype, device):
    src = tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype,
                 device)

    # src = tensor([1, 2, 3, 4, 5, 6], dtype, device)
    # src.requires_grad_()

    indptr = tensor([0, 2, 5, 5, 6], torch.long, device)
    out = segment_csr(src, indptr, reduce='any')
    print('CSR', out)
    # out = out[0] if isinstance(out, tuple) else out

    # out.backward(torch.randn_like(out))

    index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)
    out = segment_coo(src, index, reduce='any')
    print('COO', out)
