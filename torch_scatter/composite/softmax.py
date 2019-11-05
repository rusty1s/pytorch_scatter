import torch

from torch_scatter import scatter_add, scatter_max

def scatter_log_softmax(src, index, dim=-1, dim_size=None):
    r"""
    Numerical safe log-softmax of all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = softmax(\mathrm{src}_i) = 
        \mathrm{src}_i - \mathrm{logsumexp}_j ( \mathrm{src}_j)

    where :math:`\mathrm{logsumexp}_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Compute a numerically safe log softmax operation
    from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. If set to :obj:`None`,
            the output tensor is filled with the smallest possible value of
            :obj:`src.dtype`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('log_softmax can be computed only over tensors with floating point data types.')

    max_value_per_index, _ = scatter_max(src, index, dim=dim, dim_size=dim_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter_add(
        src=recentered_scores.exp(),
        index=index,
        dim=dim,
        dim_size=dim_size
    )
    log_normalizing_constants = sum_per_index.log().gather(dim, index)

    return recentered_scores - log_normalizing_constants


def scatter_softmax(src, index, dim=-1, dim_size=None, epsilon=1e-16):
    r"""
    Numerical safe log-softmax of all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = softmax(\mathrm{src}_i) = 
        \frac{\exp(\mathrm{src}_i)}{\mathrm{logsumexp}_j ( \mathrm{src}_j)}

    where :math:`\mathrm{logsumexp}_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Compute a numerically safe softmax operation
    from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. If set to :obj:`None`,
            the output tensor is filled with the smallest possible value of
            :obj:`src.dtype`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if not torch.is_floating_point(src):
        raise ValueError('softmax can be computed only over tensors with floating point data types.')

    max_value_per_index, _ = scatter_max(src, index, dim=dim, dim_size=dim_size)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    exped_recentered_scores = recentered_scores.exp()

    sum_per_index = scatter_add(
        src=exped_recentered_scores,
        index=index,
        dim=dim,
        dim_size=dim_size
    )
    normalizing_constant = (sum_per_index + epsilon).gather(dim, index)
    return exped_recentered_scores / normalizing_constant
