from .utils import gen_output


def scatter_sub_(output, index, input, dim=0):
    """If multiple indices reference the same location, their negated
    contributions add."""
    return output.scatter_add_(dim, index, -input)


def scatter_sub(index, input, dim=0, max_index=None, fill_value=0):
    output = gen_output(index, input, dim, max_index, fill_value)
    return scatter_sub_(output, index, input, dim)
