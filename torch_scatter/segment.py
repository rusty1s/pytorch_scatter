import torch

from torch_scatter.helpers import min_value, max_value

if torch.cuda.is_available():
    from torch_scatter import segment_cuda, gather_cuda


class SegmentCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, out, dim_size, reduce):
        assert reduce in ['any', 'add', 'mean', 'min', 'max']
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

        out, arg_out = segment_cuda.segment_coo(src, index, out, reduce)

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
            if ctx.reduce == 'any' or ctx.reduce == 'add':
                grad_src = gather_cuda.gather_coo(grad_out, index,
                                                  grad_out.new_empty(src_size))
            elif ctx.reduce == 'mean':
                grad_src = gather_cuda.gather_coo(grad_out, index,
                                                  grad_out.new_empty(src_size))
                count = arg_out
                count = gather_cuda.gather_coo(
                    count, index, count.new_empty(src_size[:index.dim()]))
                grad_src.div_(count)
            elif ctx.reduce == 'min' or ctx.reduce == 'max':
                src_size[index.dim() - 1] += 1
                grad_src = grad_out.new_zeros(src_size).scatter_(
                    index.dim() - 1, arg_out, grad_out)
                grad_src = grad_src.narrow(index.dim() - 1, 0,
                                           src_size[index.dim() - 1] - 1)
        return grad_src, None, None, None


class SegmentCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indptr, out, reduce):
        assert reduce in ['any', 'add', 'mean', 'min', 'max']

        if out is not None:
            ctx.mark_dirty(out)
        ctx.reduce = reduce
        ctx.src_size = list(src.size())

        out, arg_out = segment_cuda.segment_csr(src, indptr, out, reduce)
        ctx.save_for_backward(indptr, arg_out)
        return out if arg_out is None else (out, arg_out)

    @staticmethod
    def backward(ctx, grad_out, *args):
        (indptr, arg_out), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            if ctx.reduce == 'any' or ctx.reduce == 'add':
                grad_src = gather_cuda.gather_csr(grad_out, indptr,
                                                  grad_out.new_empty(src_size))
            elif ctx.reduce == 'mean':
                grad_src = gather_cuda.gather_csr(grad_out, indptr,
                                                  grad_out.new_empty(src_size))
                indptr1 = indptr.narrow(-1, 0, indptr.size(-1) - 1)
                indptr2 = indptr.narrow(-1, 1, indptr.size(-1) - 1)
                count = (indptr2 - indptr1).to(grad_src.dtype)
                count = gather_cuda.gather_csr(
                    count, indptr, count.new_empty(src_size[:indptr.dim()]))
                grad_src.div_(count)
            elif ctx.reduce == 'min' or ctx.reduce == 'max':
                src_size[indptr.dim() - 1] += 1
                grad_src = grad_out.new_zeros(src_size).scatter_(
                    indptr.dim() - 1, arg_out, grad_out)
                grad_src = grad_src.narrow(indptr.dim() - 1, 0,
                                           src_size[indptr.dim() - 1] - 1)

        return grad_src, None, None, None


def segment_coo(src, index, out=None, dim_size=None, reduce='add'):
    return SegmentCOO.apply(src, index, out, dim_size, reduce)


def segment_csr(src, indptr, out=None, reduce='add'):
    return SegmentCSR.apply(src, indptr, out, reduce)
