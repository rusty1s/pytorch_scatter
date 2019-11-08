import torch

from . import scatter_add, scatter_max


def scatter_logsumexp(src, index, dim=-1, out=None, dim_size=None,
                      fill_value=None, eps=1e-12):
    r"""Fills :attr:`out` with the log of summed exponentials of all values
    from the :attr:`src` tensor at the indices specified in the :attr:`index`
    tensor along a given axis :attr:`dim`.
    If multiple indices reference the same location, their
    **exponential contributions add**
    (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \log \, \left( \exp(\mathrm{out}_i) + \sum_j
        \exp(\mathrm{src}_j) \right)

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

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
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`None`)
        eps (float, optional): Small value to ensure numerical stability.
            (default: :obj:`1e-12`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    max_value_per_index, _ = scatter_max(src, index, dim, out, dim_size,
                                         fill_value)
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = src - max_per_src_element
    out = (out - max_per_src_element).exp() if out is not None else None

    sum_per_index = scatter_add(recentered_scores.exp(), index, dim, out,
                                dim_size, fill_value=0)

    return torch.log(sum_per_index + eps) + max_value_per_index
