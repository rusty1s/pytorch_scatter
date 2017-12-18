from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_add_(output, index, input, dim=0):
    """If multiple indices reference the same location, their contributions
    add."""
    return scatter('add', dim, output, index, input)


def scatter_add(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_add_(output, index, input, dim)


def scatter_sub_(output, index, input, dim=0):
    """If multiple indices reference the same location, their negated
    contributions add."""
    return scatter('sub', dim, output, index, input)


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
    output_count = gen_filled_tensor(output, output.size(), fill_value=0)
    scatter('mean', dim, output, index, input, output_count)
    output_count[output_count == 0] = 1
    output /= output_count
    return output


def scatter_mean(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mean_(output, index, input, dim)


def scatter_max_(output, index, input, dim=0):
    """If multiple indices reference the same location, the maximal
    contribution gets taken."""
    output_arg = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('max', dim, output, index, input, output_arg)


def scatter_max(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_max_(output, index, input, dim)


def scatter_min_(output, index, input, dim=0):
    """If multiple indices reference the same location, the minimal
    contribution gets taken."""
    output_arg = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('min', dim, output, index, input, output_arg)


def scatter_min(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_min_(output, index, input, dim)


__all__ = [
    'scatter_add_', 'scatter_add', 'scatter_sub_', 'scatter_sub',
    'scatter_mul_', 'scatter_mul', 'scatter_div_', 'scatter_div',
    'scatter_mean_', 'scatter_mean', 'scatter_max_', 'scatter_max',
    'scatter_min_', 'scatter_min'
]
