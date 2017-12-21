from .utils import gen_output


def scatter_sub_(output, index, input, dim=0):
    """If multiple indices reference the same location, their **negated
    contributions add**."""
    return output.scatter_add_(dim, index, -input)


def scatter_sub(index, input, dim=0, size=None, fill_value=0):
    """Subtracts all values from the tensor :attr:`input` at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`. The
    output size at dimension :attr:`dim` is given by :attr:`size` and must be
    at least size `index.max(dim) - 1`. If :attr:`size` is not given, a minimal
    sized output tensor is returned. The output tensor is prefilled with the
    specified value from :attr:`fill_value`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} - \sum_j \mathrm{input}_j

    where sum is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    A more detailed explanation is described in
    :meth:`~torch_scatter.scatter_sub_`.

    Args:
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index
        size (int, optional): Output size at dimension :attr:`dim`
        fill_value (int, optional): Initial filling of output tensor

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch
        from torch_scatter import scatter_sub

    .. testcode::

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
