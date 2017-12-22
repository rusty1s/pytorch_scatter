from itertools import chain

import torch
from torch.autograd import Function

from .._ext import ffi


def has_arg(name):
    return name in ['max', 'min']


def _scatter(name, dim, *data):
    a, b, c = data[:3]

    # Assert index dimension is valid.
    assert dim >= 0 and dim < a.dim(), 'Index dimension is out of bounds'

    # Assert same dimensionality across all inputs.
    assert b.dim() == c.dim(), ('Index tensor must have same dimensions as '
                                'input tensor')
    assert a.dim() == c.dim(), ('Input tensor must have same dimensions as '
                                'output tensor')

    # Assert same tensor length across index and input.
    assert b.numel() == c.numel(), ('Index tensor must have same size as '
                                    'input tensor')

    # Assert same tensor sizes across input and output apart from `dim`.
    for d in chain(range(dim), range(dim + 1, a.dim())):
        assert a.size(d) == c.size(d), (
            'Input tensor must have same size as output tensor apart from the '
            'specified dimension')

    typename = type(data[0]).__name__.replace('Tensor', '')
    cuda = 'cuda_' if data[0].is_cuda else ''
    func = getattr(ffi, 'scatter_{}_{}{}'.format(name, cuda, typename))
    func(dim, *data)
    return (data[0], data[3]) if has_arg(name) else data[0]


def index_backward(dim, index, grad, arg):
    typename = type(grad).__name__.replace('Tensor', '')
    cuda = 'cuda_' if grad.is_cuda else ''
    func = getattr(ffi, 'index_backward_{}{}'.format(cuda, typename))
    output = grad.new(index.size()).fill_(0)
    func(dim, output, index, grad, arg)
    return output


class _Scatter(Function):
    def __init__(self, name, dim):
        super(_Scatter, self).__init__()
        self.name = name
        self.dim = dim

    def forward(self, *data):
        assert not self.needs_input_grad[1], 'Can\'t differentiate the index'

        self.mark_dirty(data[0])  # Mark output as dirty.
        self.len = len(data)  # Save number of arguments for backward step.

        _scatter(self.name, self.dim, *data)

        # `scatter_min` and `scatter_max` additionally return the `argmax`
        # respectively `argmin`. Therefore, we need to save the `arg` for the
        # backward pass.
        if has_arg(self.name):
            self.save_for_backward(data[1], data[3])
            return data[0], data[3]
        else:
            self.save_for_backward(data[1])
            return data[0]

    def backward(self, *data):
        grad_output = grad_input = None

        if self.needs_input_grad[0]:
            grad_output = data[0]

        # Different grad computation of `input` if `scatter_max` or
        # `scatter_min` was used.
        if self.needs_input_grad[2] and not has_arg(self.name):
            index, = self.saved_variables
            grad_input = data[0].gather(self.dim, index.data)

        if self.needs_input_grad[2] and has_arg(self.name):
            index, arg = self.saved_variables
            data = (index.data, data[0], arg.data)
            grad_input = index_backward(self.dim, *data)

        # Return and fill with empty grads for none-differentiable passed
        # arguments in forward pass.
        return (grad_output, None, grad_input) + (None, ) * (self.len - 3)


def scatter(name, dim, *data):
    if torch.is_tensor(data[0]):
        return _scatter(name, dim, *data)
    else:
        return _Scatter(name, dim)(*data)
