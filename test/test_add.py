from nose.tools import assert_equal

import torch
from torch.autograd import Variable
from torch_scatter._ext import ffi


class ScatterAdd(torch.autograd.Function):
    def __init__(self, dim):
        super(ScatterAdd, self).__init__()
        self.dim = dim

    def forward(self, output, index, input):
        assert not self.needs_input_grad[1], 'Can\'t differentiate the index'

        self.mark_dirty(output)
        self.save_for_backward(index)

        ffi.scatter_add_Float(output, index, input, self.dim)
        return output

    def backward(self, grad):
        index, = self.saved_variables
        grad_output = grad_input = None

        if self.needs_input_grad[0]:
            grad_output = grad
        if self.needs_input_grad[2]:
            grad_input = grad.gather(self.dim, index.data)

        return grad_output, None, grad_input


def test_scatter_add():
    input = [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]
    index = [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]
    input = torch.FloatTensor(input)
    index = torch.LongTensor(index)
    output = input.new(2, 6).fill_(0)
    expected_output = [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]

    ffi.scatter_add_Float(output, index, input, 1)
    assert_equal(output.tolist(), expected_output)

    output = Variable(output)
    index = Variable(index)
    input = Variable(input, requires_grad=True)

    # a = input * 2
    # b = output * 2
    a = input * 2
    b = output * 2
    ScatterAdd(1)(b, index, a)
    # b.scatter_add_(1, index, a)

    c = b.sum()
    c.backward()

    print(input.grad)
    print(output.grad)
