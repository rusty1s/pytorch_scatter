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
    'expected': [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]],
}, {
    'name': 'add',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 0,
    'expected': [[6, 5], [6, 8]],
}, {
    'name': 'sub',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 9,
    'expected': [[9, 9, 5, 6, 6, 9], [7, 5, 5, 9, 9, 9]],
}, {
    'name': 'sub',
    'src': [[5, 2], [2, 2], [4, 2], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 9,
    'expected': [[3, 4], [3, 5]],
}, {
    'name': 'mul',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 1,
    'expected': [[1, 1, 4, 3, 2, 0], [0, 4, 3, 1, 1, 1]],
}, {
    'name': 'mul',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 1,
    'expected': [[5, 6], [8, 15]],
}, {
    'name': 'div',
    'src': [[2, 1, 1, 4, 2], [1, 2, 1, 2, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 1,
    'expected': [[1, 1, 0.25, 0.5, 0.5, 1], [0.5, 0.25, 0.5, 1, 1, 1]],
}, {
    'name': 'div',
    'src': [[4, 2], [2, 1], [4, 2], [1, 2]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 1,
    'expected': [[0.25, 0.25], [0.125, 0.5]],
}, {
    'name': 'mean',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 0,
    'expected': [[0, 0, 4, 3, 1.5, 0], [1, 4, 2, 0, 0, 0]],
}, {
    'name': 'mean',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 0,
    'expected': [[3, 2.5], [3, 4]],
}, {
    'name': 'max',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 0,
    'expected': [[0, 0, 4, 3, 2, 0], [2, 4, 3, 0, 0, 0]],
    'expected_arg': [[-1, -1, 3, 4, 0, 1], [1, 4, 3, -1, -1, -1]],
}, {
    'name': 'max',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 0,
    'expected': [[5, 3], [4, 5]],
    'expected_arg': [[0, 3], [2, 1]],
}, {
    'name': 'min',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'dim': -1,
    'fill_value': 9,
    'expected': [[9, 9, 4, 3, 1, 0], [0, 4, 1, 9, 9, 9]],
    'expected_arg': [[-1, -1, 3, 4, 2, 1], [0, 4, 2, -1, -1, -1]],
}, {
    'name': 'min',
    'src': [[5, 2], [2, 5], [4, 3], [1, 3]],
    'index': [0, 1, 1, 0],
    'dim': 0,
    'fill_value': 9,
    'expected': [[1, 2], [2, 3]],
    'expected_arg': [[3, 0], [1, 2]],
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_forward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    expected = tensor(test['expected'], dtype, device)

    op = getattr(torch_scatter, 'scatter_{}'.format(test['name']))
    out = op(src, index, test['dim'], fill_value=test['fill_value'])

    if isinstance(out, tuple):
        assert out[0].tolist() == expected.tolist()
        assert out[1].tolist() == test['expected_arg']
    else:
        assert out.tolist() == expected.tolist()
