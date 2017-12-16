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

    c = output.sum()
    c.backward()

    # # a = input * 2
    # # b = output * 2
    # a = input * 2
    # b = output * 2
    # ScatterAdd(1)(b, index, a)
    # # b.scatter_add_(1, index, a)

    # c = b.sum()
    # c.backward()

    # print(input.grad)
    # print(output.grad)
