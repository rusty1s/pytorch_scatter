from .utils import gen_output


def scatter_sub_(output, index, input, dim=0):
    """Subtracts all values from the :attr:`input` tensor into :attr:`output`
    at the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **negated contributions add** (`cf.` :meth:`~torch_scatter.scatter_add_`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{output}_i - \sum_j \mathrm{input}_j

    where sum is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    Args:
        output (Tensor): The destination tensor
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_sub_
        input = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = torch.zeros(2, 6)
        scatter_sub_(output, index, input, dim=1)
        print(output)

    .. testoutput::

        0  0 -4 -3 -3  0
       -2 -4 -4 -0  0  0
       [torch.FloatTensor of size 2x6]
    """
    return output.scatter_add_(dim, index, -input)


def scatter_sub(index, input, dim=0, size=None, fill_value=0):
    """Subtracts all values from the :attr:`input` tensor at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`
    (`cf.` :meth:`~torch_scatter.scatter_sub_` and
    :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} - \sum_j \mathrm{input}_j

    where sum is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    Args:
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index
        size (int, optional): Output size at dimension :attr:`dim`
        fill_value (int, optional): Initial filling of output tensor

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_sub
        input = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = scatter_sub(index, input, dim=1)
        print(output)

    .. testoutput::

        0  0 -4 -3 -3  0
       -2 -4 -4  0  0  0
       [torch.FloatTensor of size 2x6]
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_sub_(output, index, input, dim)
