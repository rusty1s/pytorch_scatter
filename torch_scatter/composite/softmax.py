import torch

from torch_scatter import scatter_add, scatter_max


def scatter_softmax(src, index, dim=-1, eps=1e-12):
    r"""
    Softmax operation over all values in :attr:`src` tensor that share indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = {\textrm{softmax}(\mathrm{src})}_i =
        \frac{\exp(\mathrm{src}_i)}{\sum_j \exp(\mathrm{src}_j)}

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        eps (float, optional): Small value to ensure numerical stability.
            (default: :obj:`1e-12`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    max_value_per_index, _ = scatter_max(src, index, dim=dim, fill_value=0)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter_add(recentered_scores_exp, index, dim=dim)
    normalizing_constants = (sum_per_index + eps).gather(dim, index)

    return recentered_scores_exp / normalizing_constants


def scatter_log_softmax(src, index, dim=-1, eps=1e-12):
    r"""
    Log-softmax operation over all values in :attr:`src` tensor that share
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = {\textrm{log_softmax}(\mathrm{src})}_i =
        \log \left( \frac{\exp(\mathrm{src}_i)}{\sum_j \exp(\mathrm{src}_j)}
        \right)

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        eps (float, optional): Small value to ensure numerical stability.
            (default: :obj:`1e-12`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_log_softmax` can only be computed over '
                         'tensors with floating point data types.')

    max_value_per_index, _ = scatter_max(src, index, dim=dim, fill_value=0)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter_add(src=recentered_scores.exp(), index=index,
                                dim=dim)

    normalizing_constants = torch.log(sum_per_index + eps).gather(dim, index)

    return recentered_scores - normalizing_constants
