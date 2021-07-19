from typing import Optional, Tuple

import torch


def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_csr(src, indptr, out)


def segment_min_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_csr(src, indptr, out)


def segment_max_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_csr(src, indptr, out)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor:
    r"""
    Reduces all values from the :attr:`src` tensor into :attr:`out` within the
    ranges specified in the :attr:`indptr` tensor along the last dimension of
    :attr:`indptr`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :obj:`indptr.dim() - 1` and by the
    corresponding range index in :attr:`indptr` for dimension
    :obj:`indptr.dim() - 1`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`indptr` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_0, ..., x_{m-1}, x_m, x_{m+1}, ..., x_{n-1})` and
    :math:`(x_0, ..., x_{m-2}, y)`, respectively, then :attr:`out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_0, ..., x_{m-2}, y - 1, x_{m}, ..., x_{n-1})`.
    Moreover, the values of :attr:`indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The :attr:`indptr` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i =
        \sum_{j = \mathrm{indptr}[i]}^{\mathrm{indptr}[i+1]-1}~\mathrm{src}_j.

    Due to the use of index pointers, :meth:`segment_csr` is the fastest
    method to apply for grouped reductions.

    .. note::

        In contrast to :meth:`scatter()` and :meth:`segment_coo`, this
        operation is **fully-deterministic**.

    :param src: The source tensor.
    :param indptr: The index pointers between elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import segment_csr

        src = torch.randn(10, 6, 64)
        indptr = torch.tensor([0, 2, 5, 6])
        indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.

        out = segment_csr(src, indptr, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError


def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_csr(src, indptr, out)
