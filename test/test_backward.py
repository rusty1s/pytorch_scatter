from os import path as osp
from itertools import product

import pytest
import json
import torch
from torch.autograd import Variable as V
import torch_scatter

from .utils import tensors, Tensor

f = open(osp.join(osp.dirname(__file__), 'backward.json'), 'r')
data = json.load(f)
f.close()


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_backward_cpu(tensor, i):
    name = data[i]['name']
    index = V(torch.LongTensor(data[i]['index']))
    input = V(Tensor(tensor, data[i]['input']), requires_grad=True)
    dim = data[i]['dim']
    fill_value = data[i]['fill_value']
    grad = Tensor(tensor, data[i]['grad'])
    output = V(grad.new(grad.size()).fill_(fill_value))
    expected = Tensor(tensor, data[i]['expected'])

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    func(output, index, input, dim)
    output.backward(grad)
    assert input.grad.data.tolist() == expected.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_backward_gpu(tensor, i):  # pragma: no cover
    name = data[i]['name']
    index = V(torch.LongTensor(data[i]['index']).cuda())
    input = V(Tensor(tensor, data[i]['input']).cuda(), requires_grad=True)
    dim = data[i]['dim']
    fill_value = data[i]['fill_value']
    grad = Tensor(tensor, data[i]['grad']).cuda()
    output = V(grad.new(grad.size()).fill_(fill_value).cuda())
    expected = Tensor(tensor, data[i]['expected'])

    func = getattr(torch_scatter, 'scatter_{}_'.format(name))
    func(output, index, input, dim)
    output.backward(grad)
    assert input.grad.data.cpu().tolist() == expected.tolist()
