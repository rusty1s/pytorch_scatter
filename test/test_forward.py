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
    output = Tensor(tensor, data[i]['output'])
    index = torch.LongTensor(data[i]['index'])
    input = Tensor(tensor, data[i]['input'])
    dim = data[i]['dim']
    expected = Tensor(tensor, data[i]['expected'])

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    func(output, index, input, dim)
    assert output.tolist() == expected.tolist()

    func = getattr(torch_scatter, 'scatter_{}'.format(name))
    output = func(index, input, dim)
    assert output.tolist() == expected.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_forward_gpu(tensor, i):
    name = data[i]['name']
    output = Tensor(tensor, data[i]['output']).cuda()
    index = torch.LongTensor(data[i]['index']).cuda()
    input = Tensor(tensor, data[i]['input']).cuda()
    dim = data[i]['dim']
    expected = Tensor(tensor, data[i]['expected'])

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    func(output, index, input, dim)
    assert output.cpu().tolist() == expected.tolist()

    func = getattr(torch_scatter, 'scatter_{}'.format(name))
    output = func(index, input, dim)
    assert output.cpu().tolist() == expected.tolist()
