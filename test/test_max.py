import pytest
import torch
from torch.autograd import Variable
from torch_scatter import scatter_max_, scatter_max

from .utils import tensor_strs, Tensor


# @pytest.mark.parametrize('str', tensor_strs)
@pytest.mark.parametrize('str', ['IntTensor'])
def test_scatter_mean(str):
    input = [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]
    index = [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]
    input = Tensor(str, input)
    index = torch.LongTensor(index)
    output = input.new(2, 6).fill_(0)
    expected_output = [[0, 0, 4, 3, 2, 0], [2, 4, 3, 0, 0, 0]]
    expected_output_index = [[-1, -1, 3, 4, 0, 1], [1, 4, 3, -1, -1, -1]]

    _, output_index = scatter_max_(output, index, input, dim=1)
    assert output.tolist() == expected_output
    assert output_index.tolist() == expected_output_index

    output, output_index = scatter_max(index, input, dim=1)
    assert output.tolist() == expected_output
    assert output_index.tolist() == expected_output_index

    # output = Variable(output).fill_(0)
    # index = Variable(index)
    # input = Variable(input, requires_grad=True)
    # scatter_max_(output, index, input, dim=1)

    # grad_output = [[10, 20, 30, 40, 50, 60], [15, 25, 35, 45, 55, 65]]
    # grad_output = Tensor(str, grad_output)

    # output.backward(grad_output)
    # assert index.data.tolist() == input.grad.data.tolist()

    # output = Variable(torch.FloatTensor([0, 0, 0, 0, 0]))
    index = Variable(torch.LongTensor([3, 4, 4, 2, 1]))
    input = Variable(torch.IntTensor([1, 2, 3, 4, 5]), requires_grad=True)
    output, output_index = scatter_max(index, input)
    # print(output, output_index)
    # print(output_index)

    output.backward(torch.IntTensor([10, 20, 30, 40]))
    # print(input.grad)
