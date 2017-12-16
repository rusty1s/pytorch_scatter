from nose.tools import assert_equal

import torch
from torch_scatter._ext import scatter


def test_scatter_add():
    input = [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]
    index = [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]
    input = torch.FloatTensor(input)
    index = torch.LongTensor(index)
    output = input.new(2, 6).fill_(0)
    expected_output = [[0, 0, 4, 3, 3, 0], [2, 4, 4, 0, 0, 0]]

    scatter.scatter_add_Float(output, index, input, 1)
    assert_equal(output.tolist(), expected_output)

    n = 10000
    input = torch.rand(torch.Size([n]))
    index = (torch.rand(torch.Size([n])) * n).long()
    output = input.new(n).fill_(0)
    expected_output = input.new(n).fill_(0)
    scatter.scatter_add_Float(output, index, input, 0)
    expected_output.scatter_add_(0, index, input)

    assert_equal(output.tolist(), expected_output.tolist())
