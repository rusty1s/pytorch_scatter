from .scatter import Scatter, scatter
from .utils import gen_output


class ScatterMul(Scatter):
    def __init__(self, dim):
        super(ScatterMul, self).__init__('mul', dim)

    def save_for_backward_step(self, *data):
        output, index, input = data
        self.save_for_backward(output, index, input)

    def backward_step(self, *data):  # pragma: no cover
        grad, output, index, input = data
        return (grad * output.data).gather(self.dim, index.data) / input.data


def scatter_mul_(output, index, input, dim=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/mul.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Multiplies all values from the :attr:`input` tensor into :attr:`output` at
    the indices specified in the :attr:`index` tensor along an given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions multiply** (`cf.` :meth:`~torch_scatter.scatter_add_`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{output}_i \cdot \prod_j \mathrm{input}_j

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

        from torch_scatter import scatter_mul_
        input =     torch.Tensor([[2, 0, 3, 4, 3], [2, 3, 4, 2, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = torch.ones(2, 6)
        scatter_mul_(output, index, input, dim=1)
        print(output)

    .. testoutput::

        1  1  4  3  6  0
        6  4  8  1  1  1
       [torch.FloatTensor of size 2x6]
    """
    return scatter(ScatterMul, 'mul', dim, output, index, input)


def scatter_mul(index, input, dim=0, size=None, fill_value=1):
    r"""Multiplies all values from the :attr:`input` tensor at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`
    (`cf.` :meth:`~torch_scatter.scatter_mul_` and
    :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{output}_i = \mathrm{fill\_value} \cdot \prod_j \mathrm{input}_j

    where prod is over :math:`j` such that :math:`\mathrm{index}_j = i`.

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

        from torch_scatter import scatter_mul
        input =     torch.Tensor([[2, 0, 3, 4, 3], [2, 3, 4, 2, 4]])
        index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        output = scatter_mul(index, input, dim=1)
        print(output)

    .. testoutput::

        1  1  4  3  6  0
        6  4  8  1  1  1
       [torch.FloatTensor of size 2x6]
    """
    output = gen_output(index, input, dim, size, fill_value)
    return scatter_mul_(output, index, input, dim)
