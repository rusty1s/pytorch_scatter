from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_mean_(output, index, input, dim=0):
    """If multiple indices reference the same location, their
    contributions average."""
    num_output = gen_filled_tensor(output, output.size(), fill_value=0)
    scatter('mean', dim, output, index, input, num_output)
    num_output[num_output == 0] = 1
    output /= num_output
    return output


def scatter_mean(index, input, dim=0, size=None, fill_value=0):
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_mean_(output, index, input, dim)
