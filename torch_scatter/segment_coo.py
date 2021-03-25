from typing import Optional, Tuple

import torch


def segment_sum_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_coo(src, index, out, dim_size)


def segment_add_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_coo(src, index, out, dim_size)


def segment_mean_coo(src: torch.Tensor, index: torch.Tensor,
                     out: Optional[torch.Tensor] = None,
                     dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_coo(src, index, out, dim_size)


def segment_min_coo(
        src: torch.Tensor, index: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_coo(src, index, out, dim_size)


def segment_max_coo(
        src: torch.Tensor, index: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_coo(src, index, out, dim_size)


def segment_coo(src: torch.Tensor, index: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                reduce: str = "sum") -> torch.Tensor:
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/segment_coo.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along the last dimension of
    :attr:`index`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :obj:`index.dim() - 1` and by the
    corresponding value in :attr:`index` for dimension :obj:`index.dim() - 1`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_0, ..., x_{m-1}, x_m, x_{m+1}, ..., x_{n-1})` and
    :math:`(x_0, ..., x_{m-1}, x_m)`, respectively, then :attr:`out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_0, ..., x_{m-1}, y, x_{m+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1` in ascending order.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    In contrast to :meth:`scatter`, this method expects values in :attr:`index`
    **to be sorted** along dimension :obj:`index.dim() - 1`.
    Due to the use of sorted indices, :meth:`segment_coo` is usually faster
    than the more general :meth:`scatter` operation.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The sorted indices of elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :obj:`index.dim() - 1`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import segment_coo

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 0, 1, 1, 1, 2])
        index = index.view(1, -1)  # Broadcasting in the first and last dim.

        out = segment_coo(src, index, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_coo(src, index, out, dim_size)
    elif reduce == 'mean':
        return segment_mean_coo(src, index, out, dim_size)
    elif reduce == 'min':
        return segment_min_coo(src, index, out, dim_size)[0]
    elif reduce == 'max':
        return segment_max_coo(src, index, out, dim_size)[0]
    else:
        raise ValueError


def gather_coo(src: torch.Tensor, index: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_coo(src, index, out)
