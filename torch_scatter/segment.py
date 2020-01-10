import torch

from torch_scatter import segment_cpu, gather_cpu
from torch_scatter.helpers import min_value, max_value

if torch.cuda.is_available():
    from torch_scatter import segment_cuda, gather_cuda

seg = lambda is_cuda: segment_cuda if is_cuda else segment_cpu  # noqa
gat = lambda is_cuda: gather_cuda if is_cuda else gather_cpu  # noqa


class SegmentCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, out, dim_size, reduce):
        assert reduce in ['add', 'mean', 'min', 'max']
        if out is not None:
            ctx.mark_dirty(out)
        ctx.reduce = reduce
        ctx.src_size = list(src.size())

        fill_value = 0
        if out is None:
            dim_size = index.max().item() + 1 if dim_size is None else dim_size
            size = list(src.size())
            size[index.dim() - 1] = dim_size

            if reduce == 'min':
                fill_value = max_value(src.dtype)
            elif reduce == 'max':
                fill_value = min_value(src.dtype)

            out = src.new_full(size, fill_value)

        out, arg_out = seg(src.is_cuda).segment_coo(src, index, out, reduce)

        if fill_value != 0:
            out.masked_fill_(out == fill_value, 0)

        ctx.save_for_backward(index, arg_out)

        if reduce == 'min' or reduce == 'max':
            return out, arg_out
        else:
            return out

    @staticmethod
    def backward(ctx, grad_out, *args):
        (index, arg_out), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            if ctx.reduce == 'add':
                grad_src = gat(grad_out).gather_coo(
                    grad_out, index, grad_out.new_empty(src_size))
            elif ctx.reduce == 'mean':
                grad_src = gat(grad_out).gather_coo(
                    grad_out, index, grad_out.new_empty(src_size))
                count = arg_out
                count = gat(grad_out.is_cuda).gather_coo(
                    count, index, count.new_empty(src_size[:index.dim()]))
                for _ in range(grad_out.dim() - index.dim()):
                    count = count.unsqueeze(-1)
                grad_src.div_(count)
            elif ctx.reduce == 'min' or ctx.reduce == 'max':
                src_size[index.dim() - 1] += 1
                grad_src = grad_out.new_zeros(src_size).scatter_(
                    index.dim() - 1, arg_out, grad_out)
                grad_src = grad_src.narrow(index.dim() - 1, 0,
                                           src_size[index.dim() - 1] - 1)

        return grad_src, None, None, None, None


class SegmentCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indptr, out, reduce):
        assert reduce in ['add', 'mean', 'min', 'max']

        if out is not None:
            ctx.mark_dirty(out)
        ctx.reduce = reduce
        ctx.src_size = list(src.size())

        out, arg_out = seg(src.is_cuda).segment_csr(src, indptr, out, reduce)
        ctx.save_for_backward(indptr, arg_out)
        return out if arg_out is None else (out, arg_out)

    @staticmethod
    def backward(ctx, grad_out, *args):
        (indptr, arg_out), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            if ctx.reduce == 'add':
                grad_src = gat(grad_out.is_cuda).gather_csr(
                    grad_out, indptr, grad_out.new_empty(src_size))
            elif ctx.reduce == 'mean':
                grad_src = gat(grad_out.is_cuda).gather_csr(
                    grad_out, indptr, grad_out.new_empty(src_size))
                indptr1 = indptr.narrow(-1, 0, indptr.size(-1) - 1)
                indptr2 = indptr.narrow(-1, 1, indptr.size(-1) - 1)
                count = (indptr2 - indptr1).to(grad_src.dtype)
                count = gat(grad_out.is_cuda).gather_csr(
                    count, indptr, count.new_empty(src_size[:indptr.dim()]))
                for _ in range(grad_out.dim() - indptr.dim()):
                    count = count.unsqueeze(-1)
                grad_src.div_(count)
            elif ctx.reduce == 'min' or ctx.reduce == 'max':
                src_size[indptr.dim() - 1] += 1
                grad_src = grad_out.new_zeros(src_size).scatter_(
                    indptr.dim() - 1, arg_out, grad_out)
                grad_src = grad_src.narrow(indptr.dim() - 1, 0,
                                           src_size[indptr.dim() - 1] - 1)

        return grad_src, None, None, None


def segment_coo(src, index, out=None, dim_size=None, reduce="add"):
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
    For one-dimensional tensors with :obj:`reduce="add"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    In contrast to :meth:`scatter`, this method expects values in :attr:`index`
    **to be sorted** along dimension :obj:`index.dim() - 1`.
    Due to the use of sorted indices, :meth:`segment_coo` is usually faster
    than the more general :meth:`scatter` operation.

    For reductions :obj:`"min"` and :obj:`"max"`, this operation returns a
    second tensor representing the :obj:`argmin` and :obj:`argmax`,
    respectively.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The sorted indices of elements to segment.
            The number of dimensions of :attr:`index` needs to be less than or
            equal to :attr:`src`.
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension
            :obj:`index.dim() - 1`.
            If :attr:`dim_size` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
            (default: :obj:`None`)
        reduce (string, optional): The reduce operation (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"add"`)

    :rtype: :class:`Tensor`, :class:`LongTensor` *(optional)*

    .. code-block:: python

        from torch_scatter import segment_coo

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 0, 1, 1, 1, 2])
        index = index.view(1, -1)  # Broadcasting in the first and last dim.

        out = segment_coo(src, index, reduce="add")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    return SegmentCOO.apply(src, index, out, dim_size, reduce)


def segment_csr(src, indptr, out=None, reduce="add"):
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
    :math:`(x_0, ..., x_{m-1}, y)`, respectively, then :attr:`out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_0, ..., x_{m-1}, y - 1, x_{m+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The :attr:`indptr` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors with :obj:`reduce="add"`, the operation
    computes

    .. math::
        \mathrm{out}_i =
        \sum_{j = \mathrm{indptr}[i]}^{\mathrm{indptr}[i+i]}~\mathrm{src}_j.

    Due to the use of index pointers, :meth:`segment_csr` is the fastest
    method to apply for grouped reductions.

    For reductions :obj:`"min"` and :obj:`"max"`, this operation returns a
    second tensor representing the :obj:`argmin` and :obj:`argmax`,
    respectively.

    .. note::

        In contrast to :meth:`scatter()` and :meth:`segment_coo`, this
        operation is **fully-deterministic**.

    Args:
        src (Tensor): The source tensor.
        indptr (LongTensor): The index pointers between elements to segment.
            The number of dimensions of :attr:`index` needs to be less than or
            equal to :attr:`src`.
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        reduce (string, optional): The reduce operation (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"add"`)

    :rtype: :class:`Tensor`, :class:`LongTensor` *(optional)*

    .. code-block:: python

        from torch_scatter import segment_csr

        src = torch.randn(10, 6, 64)
        indptr = torch.tensor([0, 2, 5, 6])
        indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.

        out = segment_csr(src, indptr, reduce="add")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    return SegmentCSR.apply(src, indptr, out, reduce)
