import torch

from torch_scatter import scatter_add
from torch_scatter.utils.gen import gen


def scatter_std(src, index, dim=-1, out=None, dim_size=None, unbiased=True):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/std.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Computes the standard-deviation from all values from the :attr:`src` tensor
    into :attr:`out` at the indices specified in the :attr:`index` tensor along
    a given axis :attr:`dim` (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \sqrt{\frac{\sum_j {\left( x_j - \overline{x}_i
        \right)}^2}{N_i - 1}}

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`. :math:`N_i` and :math:`\overline{x}_i`
    indicate the number of indices referencing :math:`i` and their mean value,
    respectively.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        unbiased (bool, optional): If set to :obj:`False`, then the standard-
            deviation will be calculated via the biased estimator.
            (default: :obj:`True`)

    :rtype: :class:`Tensor`
    """
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value=0)

    tmp = None if out is None else out.clone().fill_(0)
    tmp = scatter_add(src, index, dim, tmp, dim_size)

    count = None if out is None else out.clone().fill_(0)
    count = scatter_add(torch.ones_like(src), index, dim, count, dim_size)

    mean = tmp / count.clamp(min=1)

    var = (src - mean.gather(dim, index))
    var = var * var
    out = scatter_add(var, index, dim, out, dim_size)
    out = out / (count - 1 if unbiased else count).clamp(min=1)
    out = torch.sqrt(out)

    return out
