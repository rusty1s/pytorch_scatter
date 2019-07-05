import torch
from torch.autograd import Function

from torch_scatter.utils.ext import get_func
from torch_scatter.utils.gen import gen


class ScatterMax(Function):
    @staticmethod
    def forward(ctx, out, src, index, dim):
        arg = index.new_full(out.size(), -1)
        func = get_func('scatter_max', src)
        func(src, index, out, arg, dim)

        ctx.mark_dirty(out)
        ctx.dim = dim
        ctx.save_for_backward(index, arg)

        return out, arg

    @staticmethod
    def backward(ctx, grad_out, grad_arg):
        index, arg = ctx.saved_tensors

        grad_src = None
        if ctx.needs_input_grad[1]:
            size = list(index.size())
            size[ctx.dim] += 1
            grad_src = grad_out.new_zeros(size)
            grad_src.scatter_(ctx.dim, arg.detach() + 1, grad_out)
            grad_src = grad_src.narrow(ctx.dim, 1, index.size(ctx.dim))

        return None, grad_src, None, None


def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/max.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Maximizes all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.If multiple indices reference the same location, their
    **contributions maximize** (`cf.` :meth:`~torch_scatter.scatter_add`).
    The second return tensor contains index location in :attr:`src` of each
    maximum value (known as argmax).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \max(\mathrm{out}_i, \max_j(\mathrm{src}_j))

    where :math:`\max_j` is over :math:`j` such that
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
            fill output tensor with :attr:`fill_value`. If set to :obj:`None`,
            the output tensor is filled with the smallest possible value of
            :obj:`src.dtype`. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`LongTensor`)

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_max

        src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out, argmax = scatter_max(src, index, out=out)

        print(out)
        print(argmax)

    .. testoutput::

       tensor([[0., 0., 4., 3., 2., 0.],
               [2., 4., 3., 0., 0., 0.]])
       tensor([[-1, -1,  3,  4,  0,  1],
               [ 1,  4,  3, -1, -1, -1]])
    """
    if fill_value is None:
        op = torch.finfo if torch.is_floating_point(src) else torch.iinfo
        fill_value = op(src.dtype).min
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    if src.size(dim) == 0:  # pragma: no cover
        return out, index.new_full(out.size(), -1)
    return ScatterMax.apply(out, src, index, dim)
