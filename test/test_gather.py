from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_scatter import gather_coo, gather_csr
from torch_scatter.testing import devices, dtypes, tensor

tests = [
    {
        'src': [1, 2, 3, 4],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'expected': [1, 1, 2, 2, 2, 4],
    },
    {
        'src': [[1, 2], [3, 4], [5, 6], [7, 8]],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'expected': [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [7, 8]]
    },
    {
        'src': [[1, 3, 5, 7], [2, 4, 6, 8]],
        'index': [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
        'indptr': [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
        'expected': [[1, 1, 3, 3, 3, 7], [2, 2, 2, 4, 4, 6]],
    },
    {
        'src': [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
        'index': [[0, 0, 1], [0, 2, 2]],
        'indptr': [[0, 2, 3, 3], [0, 1, 1, 3]],
        'expected': [[[1, 2], [1, 2], [3, 4]], [[7, 9], [12, 13], [12, 13]]],
    },
    {
        'src': [[1], [2]],
        'index': [[0, 0], [0, 0]],
        'indptr': [[0, 2], [0, 2]],
        'expected': [[1, 1], [2, 2]],
    },
    {
        'src': [[[1, 1]], [[2, 2]]],
        'index': [[0, 0], [0, 0]],
        'indptr': [[0, 2], [0, 2]],
        'expected': [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
    },
]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_forward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test['expected'], dtype, device)

    out = gather_csr(src, indptr)
    assert torch.all(out == expected)

    out = gather_coo(src, index)
    assert torch.all(out == expected)


@pytest.mark.parametrize('test,device', product(tests, devices))
def test_backward(test, device):
    src = tensor(test['src'], torch.double, device)
    src.requires_grad_()
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)

    assert gradcheck(gather_csr, (src, indptr, None)) is True
    assert gradcheck(gather_coo, (src, index, None)) is True


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_out(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test['expected'], dtype, device)

    size = list(src.size())
    size[index.dim() - 1] = index.size(-1)
    out = src.new_full(size, -2)

    gather_csr(src, indptr, out)
    assert torch.all(out == expected)

    out.fill_(-2)

    gather_coo(src, index, out)
    assert torch.all(out == expected)


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_non_contiguous(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test['expected'], dtype, device)

    if src.dim() > 1:
        src = src.transpose(0, 1).contiguous().transpose(0, 1)
    if index.dim() > 1:
        index = index.transpose(0, 1).contiguous().transpose(0, 1)
    if indptr.dim() > 1:
        indptr = indptr.transpose(0, 1).contiguous().transpose(0, 1)

    out = gather_csr(src, indptr)
    assert torch.all(out == expected)

    out = gather_coo(src, index)
    assert torch.all(out == expected)
