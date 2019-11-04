import torch

from torch_scatter.logsumexp import _scatter_logsumexp

def scatter_log_softmax(src, index, dim=-1, dim_size=None):
    r"""
    Numerical safe log-softmax of all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = softmax(\mathrm{src}_i) = \mathrm{src}_i - \mathrm{logsumexp}_j ( \mathrm{src}_j)

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
    per_index_logsumexp, recentered_src = _scatter_logsumexp(src, index, dim=dim, dim_size=dim_size)
    return recentered_src - per_index_logsumexp.gather(dim, index)


def scatter_softmax(src, index, dim=-1, dim_size=None):
    r"""
    Numerical safe log-softmax of all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`. If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = softmax(\mathrm{src}_i) = \frac{\exp(\mathrm{src}_i)}{\mathrm{logsumexp}_j ( \mathrm{src}_j)}

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
    return scatter_log_softmax(src, index, dim, dim_size).exp()
