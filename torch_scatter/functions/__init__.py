import torch
from torch.autograd import Variable

from .scatter import scatter
from .utils import gen_output


def scatter_add_(output, index, input, dim=0):
    scatter('add', dim, output, index, input)
    return output


def scatter_add(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_add_(output, index, input, dim)


def scatter_sub_(output, index, input, dim=0):
    scatter('sub', dim, output, index, input)
    return output


def scatter_sub(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_sub_(output, index, input, dim)


def scatter_mul_(output, index, input, dim=0):
    scatter('mul', dim, output, index, input)
    return output


def scatter_mul(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mul_(output, index, input, dim)


def scatter_div_(output, index, input, dim=0):
    scatter('div', dim, output, index, input)
    return output


def scatter_div(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    scatter_div_(output, index, input, dim)


def scatter_mean_(output, index, input, dim=0):
    if torch.is_tensor(input):
        output_count = output.new(output.size()).fill_(0)
    else:
        output_count = Variable(output.data.new(output.size()).fill_(0))
    scatter('mean', dim, output, index, input, output_count)
    output_count[output_count == 0] = 1
    output /= output_count
    return output


def scatter_mean(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mean_(output, index, input, dim)


def scatter_max_(output, index, input, dim=0):
    output_index = index.new(output.size()).fill_(-1)
    scatter('max', dim, output, index, input, output_index)
    return output, output_index


def scatter_max(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_max_(output, index, input, dim)


def scatter_min_(output, index, input, dim=0):
    output_index = index.new(output.size()).fill_(-1)
    scatter('min', dim, output, index, input, output_index)
    return output, output_index


def scatter_min(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_min_(output, index, input, dim)


__all__ = [
    'scatter_add_', 'scatter_add', 'scatter_sub_', 'scatter_sub',
    'scatter_mul_', 'scatter_mul', 'scatter_div_', 'scatter_div',
    'scatter_mean_', 'scatter_mean', 'scatter_max_', 'scatter_max',
    'scatter_min_', 'scatter_min'
]
