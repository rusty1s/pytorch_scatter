from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_max_(output, index, input, dim=0):
    """If multiple indices reference the same location, the maximal
    contribution gets taken.

    :rtype: (:class:`Tensor`, :class:`LongTensor`)
    """
    arg_output = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('max', dim, output, index, input, arg_output)


def scatter_max(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_max_(output, index, input, dim)
