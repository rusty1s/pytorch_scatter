from itertools import product

import pytest
import torch
from torch_scatter import scatter, segment_coo, gather_coo
from torch_scatter import segment_csr, gather_csr

from .utils import reductions, tensor, grad_dtypes, devices


@pytest.mark.parametrize('reduce,dtype,device',
                         product(reductions, grad_dtypes, devices))
def test_zero_elements(reduce, dtype, device):
    x = torch.randn(0, 0, 0, 16, dtype=dtype, device=device,
                    requires_grad=True)
    index = tensor([], torch.long, device)
    indptr = tensor([], torch.long, device)

    out = scatter(x, index, dim=0, dim_size=0, reduce=reduce)
    out.backward(torch.randn_like(out))
    assert out.size() == (0, 0, 0, 16)

    out = segment_coo(x, index, dim_size=0, reduce=reduce)
    out.backward(torch.randn_like(out))
    assert out.size() == (0, 0, 0, 16)

    out = gather_coo(x, index)
    out.backward(torch.randn_like(out))
    assert out.size() == (0, 0, 0, 16)

    out = segment_csr(x, indptr, reduce=reduce)
    out.backward(torch.randn_like(out))
    assert out.size() == (0, 0, 0, 16)

    out = gather_csr(x, indptr)
    out.backward(torch.randn_like(out))
    assert out.size() == (0, 0, 0, 16)
