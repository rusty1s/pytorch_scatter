from .scatter import scatter
from .utils import gen_output


def scatter_div_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    **contributions divide**."""
    return scatter('div', dim, output, index, input)


def scatter_div(index, input, dim=0, size=None, fill_value=1):
    output = gen_output(index, input, dim, size, fill_value)
    scatter_div_(output, index, input, dim)
