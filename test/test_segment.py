from itertools import product

import pytest
import torch
from torch_scatter import segment_coo, segment_csr

from .utils import tensor

reductions = ['add', 'mean', 'min', 'max']

dtypes = [torch.float]
devices = [torch.device('cuda')]

tests = [
    {
        'src': [1, 2, 3, 4, 5, 6],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'add': [3, 12, 0, 6],
        'mean': [1.5, 4, 0, 6],
        'min': [1, 3, 0, 6],
        'arg_min': [0, 2, 6, 5],
        'max': [2, 5, 0, 6],
        'arg_max': [1, 4, 6, 5],
    },
    {
        'src': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'add': [[4, 6], [21, 24], [0, 0], [11, 12]],
        'mean': [[2, 3], [7, 8], [0, 0], [11, 12]],
        'min': [[1, 2], [5, 6], [0, 0], [11, 12]],
        'arg_min': [[0, 0], [2, 2], [6, 6], [5, 5]],
        'max': [[3, 4], [9, 10], [0, 0], [11, 12]],
        'arg_max': [[1, 1], [4, 4], [6, 6], [5, 5]],
    },
    {
        'src': [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
        'index': [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
        'indptr': [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
        'add': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'mean': [[2, 7, 0, 11], [4, 9, 12, 0]],
        'min': [[1, 5, 0, 11], [2, 8, 12, 0]],
        'arg_min': [[0, 2, 6, 5], [0, 3, 5, 6]],
        'max': [[3, 9, 0, 11], [6, 10, 12, 0]],
        'arg_max': [[1, 4, 6, 5], [2, 4, 5, 6]],
    },
    {
        'src': [[[1, 3, 5], [2, 4, 6]], [[7, 9, 11], [8, 10, 12]]],
        'index': [[[0, 0, 1], [0, 2, 2]], [[0, 0, 1], [0, 2, 2]]],
        'indptr': [[[0, 2, 3, 3], [0, 1, 1, 3]], [[0, 2, 3, 3], [0, 1, 1, 3]]],
        'add': [[[4, 5, 0], [2, 0, 10]], [[16, 11, 0], [8, 0, 22]]],
        'mean': [[[2, 5, 0], [2, 0, 5]], [[8, 11, 0], [8, 0, 11]]],
        'min': [[[1, 5, 0], [2, 0, 4]], [[7, 11, 0], [8, 0, 10]]],
        'arg_min': [[[0, 2, 3], [0, 3, 1]], [[0, 2, 3], [0, 3, 1]]],
        'max': [[[3, 5, 0], [2, 0, 6]], [[9, 11, 0], [8, 0, 12]]],
        'arg_max': [[[1, 2, 3], [0, 3, 2]], [[1, 2, 3], [0, 3, 2]]],
    },
    {
        'src': [[1, 3], [2, 4]],
        'index': [[0, 0], [0, 0]],
        'indptr': [[0, 2], [0, 2]],
        'add': [[4], [6]],
        'mean': [[2], [3]],
        'min': [[1], [2]],
        'arg_min': [[0], [0]],
        'max': [[3], [4]],
        'arg_max': [[1], [1]],
    },
    {
        'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
        'index': [[0, 0], [0, 0]],
        'indptr': [[0, 2], [0, 2]],
        'add': [[[4, 4]], [[6, 6]]],
        'mean': [[[2, 2]], [[3, 3]]],
        'min': [[[1, 1]], [[2, 2]]],
        'arg_min': [[[0, 0]], [[0, 0]]],
        'max': [[[3, 3]], [[4, 4]]],
        'arg_max': [[[1, 1]], [[1, 1]]],
    },
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_segment(test, reduce, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test[reduce], dtype, device)

    out = segment_coo(src, index, reduce=reduce)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test[f'arg_{reduce}'], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)

    out = segment_csr(src, indptr, reduce=reduce)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test[f'arg_{reduce}'], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_segment_out(test, reduce, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test[reduce], dtype, device)

    size = list(src.size())
    size[indptr.dim() - 1] = indptr.size(-1) - 1
    out = src.new_full(size, -2)

    # Pre-defined `out` values shouldn't do anything.
    out = segment_csr(src, indptr, out, reduce=reduce)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test[f'arg_{reduce}'], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)

    out.fill_(-2)

    out = segment_coo(src, index, out, reduce=reduce)
    out = out[0] if isinstance(out, tuple) else out

    if reduce == 'add':
        expected = expected - 2
    elif reduce == 'mean':
        expected = out  # We can not really test this here.
    elif reduce == 'min':
        expected = expected.fill_(-2)
    elif reduce == 'max':
        expected[expected == 0] = -2
    else:
        raise ValueError

    assert torch.all(out == expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_non_contiguous_segment(test, reduce, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test[reduce], dtype, device)

    if src.dim() > 1:
        src = src.transpose(0, 1).contiguous().transpose(0, 1)
    if index.dim() > 1:
        index = index.transpose(0, 1).contiguous().transpose(0, 1)
    if indptr.dim() > 1:
        indptr = indptr.transpose(0, 1).contiguous().transpose(0, 1)

    out = segment_coo(src, index, reduce=reduce)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test[f'arg_{reduce}'], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)

    out = segment_csr(src, indptr, reduce=reduce)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test[f'arg_{reduce}'], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)
