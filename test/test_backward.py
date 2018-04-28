from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
import torch_scatter

from .utils import devices

funcs = ['add', 'sub', 'mul', 'mean']
indices = [2, 0, 1, 1, 0]


@pytest.mark.parametrize('func,device', product(funcs, devices))
def test_backward(func, device):
    index = torch.tensor(indices, dtype=torch.long, device=device)
    src = torch.rand(index.size(), dtype=torch.double, device=device)
    src.requires_grad_()

    op = getattr(torch_scatter, 'scatter_{}'.format(func))
    data = (src, index)
    assert gradcheck(op, data, eps=1e-6, atol=1e-4) is True
