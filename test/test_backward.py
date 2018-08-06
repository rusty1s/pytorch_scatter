from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
import torch_scatter

from .utils import grad_dtypes as dtypes, devices, tensor

funcs = ['add', 'sub', 'mul', 'div', 'mean']
indices = [2, 0, 1, 1, 0]


@pytest.mark.parametrize('func,device', product(funcs, devices))
def test_backward(func, device):
    index = torch.tensor(indices, dtype=torch.long, device=device)
    src = torch.rand((index.size(0), 2), dtype=torch.double, device=device)
    src.requires_grad_()

    op = getattr(torch_scatter, 'scatter_{}'.format(func))
    data = (src, index, 0)
    assert gradcheck(op, data, eps=1e-6, atol=1e-4) is True


tests = [{
    'name': 'max',
    'src': [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
    'index': [2, 0, 1, 1, 0],
    'dim': 0,
    'fill_value': 0,
    'grad': [[4, 4], [8, 8], [6, 6]],
    'expected': [[6, 6], [0, 0], [0, 0], [8, 8], [4, 4]],
}, {
    'name': 'min',
    'src': [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
    'index': [2, 0, 1, 1, 0],
    'dim': 0,
    'fill_value': 3,
    'grad': [[4, 4], [8, 8], [6, 6]],
    'expected': [[6, 6], [4, 4], [8, 8], [0, 0], [0, 0]],
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_arg_backward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    src.requires_grad_()
    index = tensor(test['index'], torch.long, device)
    grad = tensor(test['grad'], dtype, device)

    op = getattr(torch_scatter, 'scatter_{}'.format(test['name']))
    out, _ = op(src, index, test['dim'], fill_value=test['fill_value'])
    out.backward(grad)
    assert src.grad.tolist() == test['expected']
