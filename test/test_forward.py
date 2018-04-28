from itertools import product

import pytest
import torch
import torch_scatter

from .utils import dtypes, devices, tensor

tests = [{
    'name': 'add',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 0,
    'expected': [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]
}, {
    'name': 'add',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [[0, 0], [1, 1], [1, 1], [0, 0]],
    'dim': 0,
    'fill_value': 0,
    'expected': [[6, 5], [6, 8]]
}, {
    'name': 'sub',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 9,
    'expected': [[9, 9, 5, 6, 6, 9], [7, 5, 5, 9, 9, 9]]
}, {
    'name': 'sub',
    'src': [[5, 2], [2, 2], [4, 2], [1, 3]],
    'index': [[0, 0], [1, 1], [1, 1], [0, 0]],
    'dim': 0,
    'fill_value': 9,
    'expected': [[3, 4], [3, 5]]
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_forward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)

    op = getattr(torch_scatter, 'scatter_{}'.format(test['name']))
    output = op(src, index, test['dim'], fill_value=test['fill_value'])

    assert output.tolist() == test['expected']
