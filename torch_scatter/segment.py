import torch

if torch.cuda.is_available():
    from torch_scatter import segment_cuda


class SegmentCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indptr, out, reduce):
        assert reduce in ['any', 'add', 'mean', 'min', 'max']
        assert indptr.dtype == torch.long

        if out is not None:
            ctx.mark_dirty(out)
        ctx.reduce = reduce
        ctx.save_for_backward(src, indptr)

        out, arg_out = segment_cuda.segment_csr(src, indptr, out, reduce)
        return out if arg_out is None else (out, arg_out)

    @staticmethod
    def backward(ctx, grad_out, *args):
        src, indptr = ctx.saved_tensors

        grad_src = None
        if ctx.needs_input_grad[0]:
            grad_src = src

        return grad_src, None, None, None


def segment_coo(src, index, out=None, dim_size=None, reduce='add'):
    assert reduce in ['any', 'add', 'mean', 'min', 'max']
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        size = list(src.size())
        size[index.dim() - 1] = dim_size
        out = src.new_zeros(size)  # TODO: DEPENDS ON REDUCE
    assert index.dtype == torch.long and src.dtype == out.dtype
    out, arg_out = segment_cuda.segment_coo(src, index, out, reduce)
    return out if arg_out is None else (out, arg_out)


def segment_csr(src, indptr, out=None, reduce='add'):
    return SegmentCSR.apply(src, indptr, out, reduce)


#     assert reduce in ['add', 'mean', 'min', 'max']
#     assert indptr.dtype == torch.long
#     out, arg_out = segment_cuda.segment_csr(src, indptr, out, reduce)
#     return out if arg_out is None else (out, arg_out)
