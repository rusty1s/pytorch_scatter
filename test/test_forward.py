from itertools import product

import pytest
import torch
import torch_scatter

from .utils import dtypes, devices, tensor

tests = [{
    'name': 'add',
    'src': [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]],
    'index': [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]],
    'fill_value': 0,
    'expected': [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_forward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)

    op = getattr(torch_scatter, 'scatter_{}'.format(test['name']))
    output = op(src, index, fill_value=test['fill_value'])

    assert output.tolist() == test['expected']
    # name = data[i]['name']
    # index = torch.LongTensor(data[i]['index'])
    # input = Tensor(tensor, data[i]['input'])
    # dim = data[i]['dim']
    # fill_value = data[i]['fill_value']
    # expected = torch.FloatTensor(data[i]['expected']).type_as(input)
    # output = expected.new(expected.size()).fill_(fill_value)

    # func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    # result = func(output, index, input, dim)
    # assert output.tolist() == expected.tolist()
    # if 'expected_arg' in data[i]:
    #     expected_arg = torch.LongTensor(data[i]['expected_arg'])
    #     assert result[1].tolist() == expected_arg.tolist()

    # func = getattr(torch_scatter, 'scatter_{}'.format(name))
    # result = func(index, input, dim, fill_value=fill_value)
    # if 'expected_arg' not in data[i]:
    #     assert result.tolist() == expected.tolist()
    # else:
    #     expected_arg = torch.LongTensor(data[i]['expected_arg'])
    #     assert result[0].tolist() == expected.tolist()
    #     assert result[1].tolist() == expected_arg.tolist()
