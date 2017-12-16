import torch
from torch.autograd import Function

from .._ext import ffi


def _scatter(name, output, index, input, dim):
    typename = type(input).__name__.replace('Tensor', '')
    func = getattr(ffi, 'scatter_{}_{}'.format(name, typename))
    func(output, index, input, dim)
    return output


class _Scatter(Function):
    def __init__(self, name, dim):
        super(_Scatter, self).__init__()
        self.dim = dim
        self.name = name

    def forward(self, output, index, input):
        assert not self.needs_input_grad[1], 'Can\'t differentiate the index'

        self.mark_dirty(output)
        self.save_for_backward(index)

        return _scatter(self.name, output, index, input, self.dim)

    def backward(self, grad):
        index, = self.saved_variables
        grad_output = grad_input = None

        if self.needs_input_grad[0]:
            grad_output = grad
        if self.needs_input_grad[2]:
            grad_input = grad.gather(self.dim, index.data)

        return grad_output, None, grad_input


def scatter(name, output, index, input, dim):
    if torch.is_tensor(input):
        return _scatter(name, output, index, input, dim)
    else:
        return _Scatter(name, dim)(output, index, input)
