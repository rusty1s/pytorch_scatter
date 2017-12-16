from .scatter import scatter
from .utils import gen_output


def scatter_add_(output, index, input, dim=0):
    return scatter('add', output, index, input, dim)


def scatter_add(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_add_(output, index, input, dim)


def scatter_sub_(output, index, input, dim=0):
    return scatter('sub', output, index, input, dim)


def scatter_sub(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_sub_(output, index, input, dim)


def scatter_mul_(output, index, input, dim=0):
    return scatter('mul', output, index, input, dim)


def scatter_mul(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mul_(output, index, input, dim)


def scatter_div_(output, index, input, dim=0):
    return scatter('div', output, index, input, dim)


def scatter_div(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_div_(output, index, input, dim)


__all__ = [
    'scatter_add_', 'scatter_add', 'scatter_sub_', 'scatter_sub',
    'scatter_mul_', 'scatter_mul', 'scatter_div_', 'scatter_div'
]
