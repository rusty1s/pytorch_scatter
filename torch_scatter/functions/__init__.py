import torch

from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_add_(output, index, input, dim=0):
    """Sums up all values from the tensor :attr:`input` into :attr:`output` at
    the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. For each value in :attr:`input`, its output index is specified
    by its index in :attr:`input` for dimension != :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension = :attr:`dim`. If
    multiple indices reference the same location, their contributions add.

    If :attr:`input` and :attr:`index` are n-dimensional tensors with equal
    size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = i, then :attr:`output` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `output.size(dim) - 1`.

    Args:
        output (Tensor): The destination tensor
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int): The axis along which to index

    Example::
        >> input = torch.Tensor([[2, 0, 1, 4, 3], [0,2, 1, 3, 4]])
        >> index = torch.LongTensor([[4, 5, 2, 3], [0, 0, 2, 2, 1]])
        >> output = torch.zeros(2, 6)
        >> scatter_add_(output, index, input, dim=1)
        0  0  4  3  3  0
        2  4  4  0  0  0
        [torch.FloatTensor of size 2x6]
    """
    return output.scatter_add_(dim, index, input)


def scatter_add(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_add_(output, index, input, dim)


def scatter_sub_(output, index, input, dim=0):
    """If multiple indices reference the same location, their negated
    contributions add."""
    return output.scatter_add_(dim, index, -input)


def scatter_sub(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_sub_(output, index, input, dim)


def scatter_mul_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions multiply."""
    return scatter('mul', dim, output, index, input)


def scatter_mul(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mul_(output, index, input, dim)


def scatter_div_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions divide."""
    return scatter('div', dim, output, index, input)


def scatter_div(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    scatter_div_(output, index, input, dim)


def scatter_mean_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions average."""
    num_output = gen_filled_tensor(output, output.size(), fill_value=0)
    scatter('mean', dim, output, index, input, num_output)
    num_output[num_output == 0] = 1
    output /= num_output
    return output


def scatter_mean(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mean_(output, index, input, dim)


def scatter_max_(output, index, input, dim=0):
    """If multiple indices reference the same location, the maximal
    contribution gets taken."""
    arg_output = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('max', dim, output, index, input, arg_output)


def scatter_max(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_max_(output, index, input, dim)


def scatter_min_(output, index, input, dim=0):
    """If multiple indices reference the same location, the minimal
    contribution gets taken."""
    arg_output = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('min', dim, output, index, input, arg_output)


def scatter_min(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_min_(output, index, input, dim)


__all__ = [
    'scatter_add_', 'scatter_add', 'scatter_sub_', 'scatter_sub',
    'scatter_mul_', 'scatter_mul', 'scatter_div_', 'scatter_div',
    'scatter_mean_', 'scatter_mean', 'scatter_max_', 'scatter_max',
    'scatter_min_', 'scatter_min'
]
