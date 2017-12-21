from .scatter import scatter
from .utils import gen_output


def scatter_mul_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions multiply."""
    return scatter('mul', dim, output, index, input)


def scatter_mul(index, input, dim=0, max_index=None, fill_value=1):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_mul_(output, index, input, dim)
