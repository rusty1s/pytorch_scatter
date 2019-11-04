import torch

from . import scatter_add, scatter_max

EPSILON = 1e-16

def _scatter_logsumexp(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    if not torch.is_floating_point(src):
        raise ValueError('logsumexp can be computed over tensors floating point data types.')

    if fill_value is None:
        fill_value = torch.finfo(src.dtype).min

    dim_size = out.shape[dim] if dim_size is None and out is not None else dim_size
    max_value_per_index, _ = scatter_max(src, index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter_add(
        src=recentered_scores.exp(),
        index=index,
        dim=dim,
        out=(src - max_per_src_element).exp() if out is not None else None,
        dim_size=dim_size,
        fill_value=fill_value,
    )
    return torch.log(sum_per_index + EPSILON) + max_value_per_index, recentered_scores

def scatter_logsumexp(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    r"""
    Numerically safe logsumexp of all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions logsumexp** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \log \left( \exp(\mathrm{out}_i) + \sum_j \exp(\mathrm{src}_j) \right)

    Compute a numerically safe logsumexp operation
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
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
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
    return _scatter_logsumexp(src,index, dim, out, dim_size, fill_value)[0]
