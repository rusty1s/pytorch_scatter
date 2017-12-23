from .scatter import scatter
from .utils import gen_filled_tensor, gen_output


def scatter_mean_(output, index, input, dim=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/mean.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Averages all values from the :attr:`input` tensor into :attr:`output` at
    the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add_`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{output}_i + \frac{1}{N_i} \cdot
        \sum_j \mathrm{input}_j

    where sum is over :math:`j` such that :math:`\mathrm{index}_j = i` and
    :math:`N_i` indicates the number of indices referencing :math:`i`.

    Args:
        output (Tensor): The destination tensor
        index (LongTensor): The indices of elements to scatter
        input (Tensor): The source tensor
        dim (int, optional): The axis along which to index

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_mean_
        input =     torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = torch.zeros(2, 6)
        scatter_mean_(output, index, input, dim=1)
        print(output)

    .. testoutput::

        0.0000  0.0000  4.0000  3.0000  1.5000  0.0000
        1.0000  4.0000  2.0000  0.0000  0.0000  0.0000
       [torch.FloatTensor of size 2x6]
    """
    count = gen_filled_tensor(output, output.size(), fill_value=0)
    scatter('mean', dim, output, index, input, count)
    count[count == 0] = 1
    output /= count
    return output


def scatter_mean(index, input, dim=0, size=None, fill_value=0):
    r"""Averages all values from the :attr:`input` tensor at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`
    (`cf.` :meth:`~torch_scatter.scatter_mean_` and
    :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} + \frac{1}{N_i} \cdot
        \sum_j \mathrm{input}_j

    where sum is over :math:`j` such that :math:`\mathrm{index}_j = i` and
    :math:`N_i` indicates the number of indices referencing :math:`i`.

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

        from torch_scatter import scatter_mean
        input =     torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = scatter_mean(index, input, dim=1)
        print(output)

    .. testoutput::

        0.0000  0.0000  4.0000  3.0000  1.5000  0.0000
        1.0000  4.0000  2.0000  0.0000  0.0000  0.0000
       [torch.FloatTensor of size 2x6]
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_mean_(output, index, input, dim)
