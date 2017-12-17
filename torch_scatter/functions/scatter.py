import torch
from torch.autograd import Function

from .._ext import ffi


def _scatter(name, dim, *data):
    typename = type(data[0]).__name__.replace('Tensor', '')
    func = getattr(ffi, 'scatter_{}_{}'.format(name, typename))
    func(dim, *data)


class _Scatter(Function):
    def __init__(self, name, dim):
        super(_Scatter, self).__init__()
        self.name = name
        self.dim = dim

    def forward(self, *data):
        assert not self.needs_input_grad[1], 'Can\'t differentiate the index'

        self.mark_dirty(data[0])  # Mark output as dirty.
        self.len = len(data)  # Save number of arguments for backward step
        self.save_for_backward(data[1])  # Save index for backward step.

        _scatter(self.name, self.dim, *data)
        return data[0]

    def backward(self, *data):
        index, = self.saved_variables
        grad_output = grad_input = None

        if self.needs_input_grad[0]:
            grad_output = data[0]
        if self.needs_input_grad[2]:
            grad_input = data[0].gather(self.dim, index.data)

        return (grad_output, None, grad_input) + (None, ) * (self.len - 3)


def scatter(name, dim, *data):
    if torch.is_tensor(data[0]):
        return _scatter(name, dim, *data)
    else:
        return _Scatter(name, dim)(*data)
