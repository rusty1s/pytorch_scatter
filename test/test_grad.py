import torch
from torch.autograd import Variable


def test_grad():
    input = [[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]
    index = [[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]
    input = torch.FloatTensor(input)
    index = torch.LongTensor(index)
    output = input.new(2, 6).fill_(0)

    output = Variable(output)
    index = Variable(index)
    input = Variable(input, requires_grad=True)

    output.scatter_add_(1, index, input)

    c = output.mean()
    c.backward()
    # print(index.grad)
