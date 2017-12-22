from .scatter import scatter
from .utils import gen_output


def scatter_div_(output, index, input, dim=0):
    r"""Divides all values from the :attr:`input` tensor into :attr:`output`
    at the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions divide** (`cf.` :meth:`~torch_scatter.scatter_add_`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{output}_i \cdot \prod_j
        \frac{1}{\mathrm{input}_j}

    where prod is over :math:`j` such that :math:`\mathrm{index}_j = i`.

    Args:
        output (Tensor): The destination tensor
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_div_
        input =     torch.Tensor([[2, 1, 2, 4, 3], [1, 2, 2, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = torch.ones(2, 6)
        scatter_div_(output, index, input, dim=1)
        print(output)

    .. testoutput::

        1.0000  1.0000  0.2500  0.3333  0.2500  1.0000
        0.5000  0.2500  0.1667  1.0000  1.0000  1.0000
       [torch.FloatTensor of size 2x6]
    """
    return scatter('div', dim, output, index, input)


def scatter_div(index, input, dim=0, size=None, fill_value=1):
    r"""Divides all values from the :attr:`input` tensor at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`
    (`cf.` :meth:`~torch_scatter.scatter_div_` and
    :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} \cdot \prod_j
        \frac{1}{\mathrm{input}_j}

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

        from torch_scatter import scatter_div
        input =     torch.Tensor([[2, 1, 2, 4, 3], [1, 2, 2, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = scatter_div(index, input, dim=1)
        print(output)

    .. testoutput::

        1.0000  1.0000  0.2500  0.3333  0.2500  1.0000
        0.5000  0.2500  0.1667  1.0000  1.0000  1.0000
       [torch.FloatTensor of size 2x6]
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_div_(output, index, input, dim)
