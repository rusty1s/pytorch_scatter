from .scatter import scatter
from .utils import gen_output


def scatter_div_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions divide."""
    return scatter('div', dim, output, index, input)


def scatter_div(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    scatter_div_(output, index, input, dim)
