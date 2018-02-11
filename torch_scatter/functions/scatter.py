import torch
from torch.autograd import Function

from .ffi import scatter as ffi_scatter


class Scatter(Function):
    def __init__(self, name, dim):
        super(Scatter, self).__init__()
        self.name = name
        self.dim = dim

    def save_for_backward_step(self, *data):  # pragma: no cover
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

    def backward_step(self, *data):  # pragma: no cover
        raise NotImplementedError


def scatter(Clx, name, dim, *data):
    if torch.is_tensor(data[0]):
        return ffi_scatter(name, dim, *data)
    else:
        return Clx(dim)(*data)
