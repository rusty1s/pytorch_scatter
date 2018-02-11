from itertools import chain

from .._ext import ffi


def scatter(name, dim, *data):
    # data = output, index, input, additional data
    a, b, c = data[:3]

    # Assert index dimension is valid.
    assert dim >= 0 and dim < b.dim(), 'Index dimension is out of bounds'

    # Assert same dimensionality across all inputs.
    assert b.dim() == c.dim(), ('Index tensor must have same dimensions as '
                                'input tensor')
    assert a.dim() == c.dim(), ('Input tensor must have same dimensions as '
                                'output tensor')

    # Assert same tensor length across index and input.
    assert b.numel() == c.numel(), ('Index tensor must have same size as '
                                    'input tensor')

    # Assert same tensor sizes across input and output apart from `dim`.
    for d in chain(range(dim), range(dim + 1, a.dim())):
        assert a.size(d) == c.size(d), (
            'Input tensor must have same size as output tensor apart from the '
            'specified dimension')

    typename = type(data[0]).__name__.replace('Tensor', '')
    cuda = 'cuda_' if data[0].is_cuda else ''
    func = getattr(ffi, 'scatter_{}_{}{}'.format(name, cuda, typename))
    func(dim, *data)

    if len(data) <= 3:
        return data[0]

    return (data[0], ) + tuple(data[3:])


def index_backward(dim, index, grad, arg):  # pragma: no cover
    typename = type(grad).__name__.replace('Tensor', '')
    cuda = 'cuda_' if grad.is_cuda else ''
    func = getattr(ffi, 'index_backward_{}{}'.format(cuda, typename))
    output = grad.new(index.size()).fill_(0)
    func(dim, output, index, grad, arg)
    return output
