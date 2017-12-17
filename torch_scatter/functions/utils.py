import torch
from torch.autograd import Variable


def gen_filled_tensor(input, size, fill_value):
    if torch.is_tensor(input):
        return input.new(size).fill_(fill_value)
    else:
        return Variable(input.data.new(size).fill_(fill_value))


def gen_output(index, input, dim, max_index, fill_value):
    max_index = index.max() + 1 if max_index is None else max_index
    size = list(index.size())

    if torch.is_tensor(input):
        size[dim] = max_index
    else:
        size[dim] = max_index.data[0]

    return gen_filled_tensor(input, torch.Size(size), fill_value)
