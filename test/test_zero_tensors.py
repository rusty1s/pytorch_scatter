from itertools import product

import pytest
import torch
from torch_scatter import (gather_coo, gather_csr, scatter, segment_coo,
                           segment_csr)
from torch_scatter.testing import devices, grad_dtypes, reductions, tensor


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
