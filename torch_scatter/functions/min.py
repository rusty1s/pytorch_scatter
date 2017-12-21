from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_min_(output, index, input, dim=0):
    """If multiple indices reference the same location, the minimal
    contribution gets taken."""
    arg_output = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('min', dim, output, index, input, arg_output)


def scatter_min(index, input, dim=0, size=None, fill_value=0):
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_min_(output, index, input, dim)
