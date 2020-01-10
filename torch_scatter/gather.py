import torch

from torch_scatter import segment_cpu, gather_cpu

if torch.cuda.is_available():
    from torch_scatter import gather_cuda, segment_cuda


def gat(is_cuda):
    return gather_cuda if is_cuda else gather_cpu


def seg(is_cuda):
    return segment_cuda if is_cuda else segment_cpu


class GatherCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, out):
        if out is not None:
            ctx.mark_dirty(out)
        ctx.src_size = list(src.size())
        ctx.save_for_backward(index)

        return gat(src.is_cuda).gather_coo(src, index, out)

    @staticmethod
    def backward(ctx, grad_out):
        (index, ), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            grad_src, _ = seg(grad_out.is_cuda).segment_coo(
                grad_out, index, grad_out.new_zeros(src_size), 'add')

        return grad_src, None, None


class GatherCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indptr, out):
        if out is not None:
            ctx.mark_dirty(out)
        ctx.src_size = list(src.size())
        ctx.save_for_backward(indptr)

        return gat(src.is_cuda).gather_csr(src, indptr, out)

    @staticmethod
    def backward(ctx, grad_out):
        (indptr, ), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            grad_src, _ = seg(grad_out.is_cuda).segment_csr(
                grad_out, indptr, grad_out.new_empty(src_size), 'add')

        return grad_src, None, None


def gather_coo(src, index, out=None):
    return GatherCOO.apply(src, index, out)


def gather_csr(src, indptr, out=None):
    return GatherCSR.apply(src, indptr, out)
