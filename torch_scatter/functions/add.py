from .utils import gen_output


def scatter_add_(output, index, input, dim=0):
    r"""
    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    Sums all values from the :attr:`input` tensor into :attr:`output` at the
    indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. For each value in :attr:`input`, its output index is specified
    by its index in :attr:`input` for dimensions outside of :attr:`dim` and by
    the corresponding value in :attr:`index` for dimension :attr:`dim`. If
    multiple indices reference the same location, their **contributions add**.

    If :attr:`input` and :attr:`index` are n-dimensional tensors with size
    :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = `i`, then :attr:`output` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `output.size(dim) - 1`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{output}_i + \sum_j \mathrm{input}_j

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

        from torch_scatter import scatter_add_
        input =     torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = torch.zeros(2, 6)
        scatter_add_(output, index, input, dim=1)
        print(output)

    .. testoutput::

        0  0  4  3  3  0
        2  4  4  0  0  0
       [torch.FloatTensor of size 2x6]
    """
    return output.scatter_add_(dim, index, input)


def scatter_add(index, input, dim=0, size=None, fill_value=0):
    r"""Sums all values from the :attr:`input` tensor at the indices specified
    in the :attr:`index` tensor along an given axis :attr:`dim` (`cf.`
    :meth:`~torch_scatter.scatter_add_`).

    The output size at dimension :attr:`dim` is given by :attr:`size` and must
    be at least size `index.max(dim) - 1`. If :attr:`size` is not given, a
    minimal sized output tensor is returned. The output tensor is prefilled
    with the specified value from :attr:`fill_value`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} + \sum_j \mathrm{input}_j

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

        from torch_scatter import scatter_add
        input =     torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = scatter_add(index, input, dim=1)
        print(output)

    .. testoutput::

        0  0  4  3  3  0
        2  4  4  0  0  0
       [torch.FloatTensor of size 2x6]
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_add_(output, index, input, dim)
