import torch
from torch.autograd import Variable


def gen_output(index, input, dim, max_index, fill_value):
    max_index = index.max() + 1 if max_index is None else max_index
    size = list(index.size())

    if torch.is_tensor(input):
        size[dim] = max_index
        return input.new(torch.Size(size)).fill_(fill_value)
    else:
        size[dim] = max_index.data[0]
        return Variable(input.data.new(torch.Size(size)).fill_(fill_value))
