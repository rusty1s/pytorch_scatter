import torch


class GatherCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, out):
        if out is not None:
            ctx.mark_dirty(out)
        ctx.src_size = list(src.size())
        ctx.save_for_backward(index)

        if src.is_cuda:
            return torch.ops.torch_scatter_cuda.gather_coo(src, index, out)
        else:
            return torch.ops.torch_scatter_cpu.gather_coo(src, index, out)

    @staticmethod
    def backward(ctx, grad_out):
        (index, ), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            if grad_out.is_cuda:
                grad_src, _ = torch.ops.torch_scatter_cuda.segment_coo(
                    grad_out, index, grad_out.new_zeros(src_size), 'sum')
            else:
                grad_src, _ = torch.ops.torch_scatter_cpu.segment_coo(
                    grad_out, index, grad_out.new_zeros(src_size), 'sum')

        return grad_src, None, None


class GatherCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indptr, out):
        if out is not None:
            ctx.mark_dirty(out)
        ctx.src_size = list(src.size())
        ctx.save_for_backward(indptr)

        if src.is_cuda:
            return torch.ops.torch_scatter_cuda.gather_csr(src, indptr, out)
        else:
            return torch.ops.torch_scatter_cpu.gather_csr(src, indptr, out)

    @staticmethod
    def backward(ctx, grad_out):
        (indptr, ), src_size = ctx.saved_tensors, ctx.src_size

        grad_src = None
        if ctx.needs_input_grad[0]:
            if grad_out.is_cuda:
                grad_src, _ = torch.ops.torch_scatter_cuda.segment_csr(
                    grad_out, indptr, grad_out.new_empty(src_size), 'sum')
            else:
                grad_src, _ = torch.ops.torch_scatter_cpu.segment_csr(
                    grad_out, indptr, grad_out.new_empty(src_size), 'sum')

        return grad_src, None, None


def gather_coo(src, index, out=None):
    return GatherCOO.apply(src, index, out)


def gather_csr(src, indptr, out=None):
    return GatherCSR.apply(src, indptr, out)
