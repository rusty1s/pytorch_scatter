from os import path as osp
from itertools import product

import pytest
import json
import torch
import torch_scatter

from .utils import tensors, Tensor

f = open(osp.join(osp.dirname(__file__), 'forward.json'), 'r')
data = json.load(f)
f.close()


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_forward_cpu(tensor, i):
    name = data[i]['name']
    index = torch.LongTensor(data[i]['index'])
    input = Tensor(tensor, data[i]['input'])
    dim = data[i]['dim']
    fill_value = data[i]['fill_value']
    expected = torch.FloatTensor(data[i]['expected']).type_as(input)
    output = expected.new(expected.size()).fill_(fill_value)

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    result = func(output, index, input, dim)
    assert output.tolist() == expected.tolist()
    if 'expected_arg' in data[i]:
        expected_arg = torch.LongTensor(data[i]['expected_arg'])
        assert result[1].tolist() == expected_arg.tolist()

    func = getattr(torch_scatter, 'scatter_{}'.format(name))
    result = func(index, input, dim, fill_value=fill_value)
    if 'expected_arg' not in data[i]:
        assert result.tolist() == expected.tolist()
    else:
        expected_arg = torch.LongTensor(data[i]['expected_arg'])
        assert result[0].tolist() == expected.tolist()
        assert result[1].tolist() == expected_arg.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_forward_gpu(tensor, i):  # pragma: no cover
    name = data[i]['name']
    index = torch.cuda.LongTensor(data[i]['index'])
    input = Tensor(tensor, data[i]['input']).cuda()
    dim = data[i]['dim']
    fill_value = data[i]['fill_value']
    expected = torch.FloatTensor(data[i]['expected']).type_as(input)
    output = expected.new(expected.size()).fill_(fill_value).cuda()

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    result = func(output, index, input, dim)
    assert output.cpu().tolist() == expected.tolist()
    if 'expected_arg' in data[i]:
        expected_arg = torch.LongTensor(data[i]['expected_arg'])
        assert result[1].cpu().tolist() == expected_arg.tolist()
    func = getattr(torch_scatter, 'scatter_{}'.format(name))
    result = func(index, input, dim, fill_value=fill_value)
    if 'expected_arg' not in data[i]:
        assert result.cpu().tolist() == expected.tolist()
    else:
        expected_arg = torch.LongTensor(data[i]['expected_arg'])
        assert result[0].cpu().tolist() == expected.tolist()
        assert result[1].cpu().tolist() == expected_arg.tolist()
