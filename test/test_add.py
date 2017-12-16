from nose.tools import assert_equal

import torch
from torch.autograd import Variable
from torch_scatter import scatter_add_, scatter_add


def test_scatter_add():
    input = [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]
    index = [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]
    input = torch.FloatTensor(input)
    index = torch.LongTensor(index)
    output = input.new(2, 6).fill_(0)
    expected_output = [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]

    scatter_add_(output, index, input, dim=1)
    assert_equal(output.tolist(), expected_output)

    output = scatter_add(index, input, dim=1)
    assert_equal(output.tolist(), expected_output)

    output = Variable(output).fill_(0)
    index = Variable(index)
    input = Variable(input, requires_grad=True)
    scatter_add_(output, index, input, dim=1)

    grad_output = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    grad_output = torch.FloatTensor(grad_output)

    output.backward(grad_output)
    assert_equal(index.data.tolist(), input.grad.data.tolist())
