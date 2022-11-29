from itertools import product

import pytest
import torch
import torch_scatter
from torch.autograd import gradcheck
from torch_scatter.testing import devices, dtypes, reductions, tensor

tests = [
    {
        'src': [1, 2, 3, 4, 5, 6],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'sum': [3, 12, 0, 6],
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
        'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
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
        'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'add': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'mean': [[2, 7, 0, 11], [4, 9, 12, 0]],
        'min': [[1, 5, 0, 11], [2, 8, 12, 0]],
        'arg_min': [[0, 2, 6, 5], [0, 3, 5, 6]],
        'max': [[3, 9, 0, 11], [6, 10, 12, 0]],
        'arg_max': [[1, 4, 6, 5], [2, 4, 5, 6]],
    },
    {
        'src': [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
        'index': [[0, 0, 1], [0, 2, 2]],
        'indptr': [[0, 2, 3, 3], [0, 1, 1, 3]],
        'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        'add': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        'mean': [[[2, 3], [5, 6], [0, 0]], [[7, 9], [0, 0], [11, 12]]],
        'min': [[[1, 2], [5, 6], [0, 0]], [[7, 9], [0, 0], [10, 11]]],
        'arg_min': [[[0, 0], [2, 2], [3, 3]], [[0, 0], [3, 3], [1, 1]]],
        'max': [[[3, 4], [5, 6], [0, 0]], [[7, 9], [0, 0], [12, 13]]],
        'arg_max': [[[1, 1], [2, 2], [3, 3]], [[0, 0], [3, 3], [2, 2]]],
    },
    {
        'src': [[1, 3], [2, 4]],
        'index': [[0, 0], [0, 0]],
        'indptr': [[0, 2], [0, 2]],
        'sum': [[4], [6]],
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
        'sum': [[[4, 4]], [[6, 6]]],
        'add': [[[4, 4]], [[6, 6]]],
        'mean': [[[2, 2]], [[3, 3]]],
        'min': [[[1, 1]], [[2, 2]]],
        'arg_min': [[[0, 0]], [[0, 0]]],
        'max': [[[3, 3]], [[4, 4]]],
        'arg_max': [[[1, 1]], [[1, 1]]],
    },
]


@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_forward(test, reduce, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test[reduce], dtype, device)

    fn = getattr(torch_scatter, 'segment_' + reduce + '_csr')
    jit = torch.jit.script(fn)
    out1 = fn(src, indptr)
    out2 = jit(src, indptr)
    if isinstance(out1, tuple):
        out1, arg_out1 = out1
        out2, arg_out2 = out2
        arg_expected = tensor(test['arg_' + reduce], torch.long, device)
        assert torch.all(arg_out1 == arg_expected)
        assert arg_out1.tolist() == arg_out2.tolist()
    assert torch.all(out1 == expected)
    assert out1.tolist() == out2.tolist()

    fn = getattr(torch_scatter, 'segment_' + reduce + '_coo')
    jit = torch.jit.script(fn)
    out1 = fn(src, index)
    out2 = jit(src, index)
    if isinstance(out1, tuple):
        out1, arg_out1 = out1
        out2, arg_out2 = out2
        arg_expected = tensor(test['arg_' + reduce], torch.long, device)
        assert torch.all(arg_out1 == arg_expected)
        assert arg_out1.tolist() == arg_out2.tolist()
    assert torch.all(out1 == expected)
    assert out1.tolist() == out2.tolist()


@pytest.mark.parametrize('test,reduce,device',
                         product(tests, reductions, devices))
def test_backward(test, reduce, device):
    src = tensor(test['src'], torch.double, device)
    src.requires_grad_()
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)

    assert gradcheck(torch_scatter.segment_csr, (src, indptr, None, reduce))
    assert gradcheck(torch_scatter.segment_coo,
                     (src, index, None, None, reduce))


@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_out(test, reduce, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    expected = tensor(test[reduce], dtype, device)

    out = torch.full_like(expected, -2)

    getattr(torch_scatter, 'segment_' + reduce + '_csr')(src, indptr, out)
    assert torch.all(out == expected)

    out.fill_(-2)

    getattr(torch_scatter, 'segment_' + reduce + '_coo')(src, index, out)

    if reduce == 'sum' or reduce == 'add':
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


@pytest.mark.parametrize('test,reduce,dtype,device',
                         product(tests, reductions, dtypes, devices))
def test_non_contiguous(test, reduce, dtype, device):
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

    out = getattr(torch_scatter, 'segment_' + reduce + '_csr')(src, indptr)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test['arg_' + reduce], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)

    out = getattr(torch_scatter, 'segment_' + reduce + '_coo')(src, index)
    if isinstance(out, tuple):
        out, arg_out = out
        arg_expected = tensor(test['arg_' + reduce], torch.long, device)
        assert torch.all(arg_out == arg_expected)
    assert torch.all(out == expected)
