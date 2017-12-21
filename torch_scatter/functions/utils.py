import torch
from torch.autograd import Variable


def gen_filled_tensor(input, size, fill_value):
    if torch.is_tensor(input):
        return input.new(size).fill_(fill_value)
    else:
        return Variable(input.data.new(size).fill_(fill_value))


def gen_output(index, input, dim, dim_size, fill_value):
    if dim_size is None:
        dim_size = index.max() + 1
        dim_size = dim_size if torch.is_tensor(input) else dim_size.data[0]

    size = list(index.size())
    size[dim] = dim_size
    return gen_filled_tensor(input, torch.Size(size), fill_value)
