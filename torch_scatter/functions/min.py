from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_min_(output, index, input, dim=0):
    r"""Minimizes all values from the :attr:`input` tensor into :attr:`output`
    at the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions minimize** (`cf.` :meth:`~torch_scatter.scatter_add_`).
    The second return value is the index location in :attr:`input` of each
    minimum value found (argmin).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \min(\mathrm{output}_i, \min_j(\mathrm{input}_j))

    where min is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    Args:
        output (Tensor): The destination tensor
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index

    :rtype: (:class:`Tensor`, :class:`LongTensor`)

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_min_
        input =     torch.Tensor([[-2, 0, -1, -4, -3], [0, -2, -1, -3, -4]])
        index = torch.LongTensor([[ 4, 5,  4,  2,  3], [0,  0,  2,  2,  1]])
        output = torch.zeros(2, 6)
        output = scatter_min_(output, index, input, dim=1)
        print(output)

    .. testoutput::

       (
        0  0 -4 -3 -2  0
       -2 -4 -3  0  0  0
       [torch.FloatTensor of size 2x6]
       ,
       -1 -1  3  4  0  1
        1  4  3 -1 -1 -1
       [torch.LongTensor of size 2x6]
       )
    """
    arg_output = gen_filled_tensor(index, output.size(), fill_value=-1)
    return scatter('min', dim, output, index, input, arg_output)


def scatter_min(index, input, dim=0, size=None, fill_value=0):
    r"""Minimizes all values from the :attr:`input` tensor at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`
    (`cf.` :meth:`~torch_scatter.scatter_min_` and
    :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \min(\mathrm{fill\_value},
        \min_j(\mathrm{input}_j))

    where min is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    Args:
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index
        size (int, optional): Output size at dimension :attr:`dim`
        fill_value (int, optional): Initial filling of output tensor

    :rtype: (:class:`Tensor`, :class:`LongTensor`)

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_min
        input =     torch.Tensor([[-2, 0, -1, -4, -3], [0, -2, -1, -3, -4]])
        index = torch.LongTensor([[ 4, 5,  4,  2,  3], [0,  0,  2,  2,  1]])
        output = scatter_min(index, input, dim=1)
        print(output)

    .. testoutput::

       (
        0  0 -4 -3 -2  0
       -2 -4 -3  0  0  0
       [torch.FloatTensor of size 2x6]
       ,
       -1 -1  3  4  0  1
        1  4  3 -1 -1 -1
       [torch.LongTensor of size 2x6]
       )
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_min_(output, index, input, dim)
