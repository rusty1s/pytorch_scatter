import torch
from torch.autograd import Function

from .ffi import scatter as ffi_scatter


class Scatter(Function):
    def __init__(self, name, dim):
        super(Scatter, self).__init__()
        self.name = name
        self.dim = dim

    def save_for_backward_step(self, *data):
        raise NotImplementedError

    def forward(self, *data):
        assert not self.needs_input_grad[1], 'Can\'t differentiate the index'

        self.mark_dirty(data[0])  # Mark output as dirty.
        self.len = len(data)  # Save number of arguments for backward step.

        output = ffi_scatter(self.name, self.dim, *data)
        self.save_for_backward_step(*data)

        return output

    def backward(self, *data):  # pragma: no cover
        grad_output = grad_input = None

        if self.needs_input_grad[0]:
            grad_output = data[0]

        # Call grad computation of `input` for the specific scatter operation.
        if self.needs_input_grad[2]:
            grad_input = self.backward_step(data[0], *self.saved_variables)

        # Return and fill with empty grads for non-differentiable arguments.
        return (grad_output, None, grad_input) + (None, ) * (self.len - 3)

    def backward_step(self, *data):
        raise NotImplementedError


def scatter(Clx, name, dim, *data):
    if torch.is_tensor(data[0]):
        return ffi_scatter(name, dim, *data)
    else:
        return Clx(dim)(*data)


# def index_backward(dim, index, grad, arg):  # pragma: no cover
#     typename = type(grad).__name__.replace('Tensor', '')
#     cuda = 'cuda_' if grad.is_cuda else ''
#     func = getattr(ffi, 'index_backward_{}{}'.format(cuda, typename))
#     output = grad.new(index.size()).fill_(0)
#     func(dim, output, index, grad, arg)
#     return output

# def _scatter_backward(name, dim, saved, *data):
#     # saved = (index, ), (index, arg) or (index, count)

#     print(name)
#     print(len(data))
#     print(len(saved))
#     print(saved[1].size())
#     # data = (grad, )
#     # index, = seved
#     if has_arg(name):
#         return index_backward(dim, saved[0].data, data[0], saved[1].data)

#     if has_count(name):
#         return (data[0] / saved[1]).gather(dim, saved[0].data)
#     # Different grad computation of `input` if `scatter_max` or
#     # `scatter_min` was used.
#     # if self.needs_input_grad[2] and not has_arg(self.name):
#     #     index, = self.saved_variables
#     #     grad_input = data[0].gather(self.dim, index.data)

#     # if self.needs_input_grad[2] and has_arg(self.name):
#     #     index, arg = self.saved_variables
#     #     data = (index.data, data[0], arg.data)
#     grad_input = index_backward(self.dim, *data)
