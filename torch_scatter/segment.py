import torch

if torch.cuda.is_available():
    from torch_scatter import segment_cuda


def segment_coo(src, index, out=None, dim_size=None, reduce='add'):
    assert reduce in ['add', 'mean', 'min', 'max']
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        size = list(src.size())
        size[index.dim() - 1] = dim_size
        out = src.new_zeros(size)  # TODO: DEPENDENT ON REDUCE
    assert index.dtype == torch.long and src.dtype == out.dtype
    out, arg_out = segment_cuda.segment_coo(src, index, out, reduce)
    return out if arg_out is None else (out, arg_out)


def segment_csr(src, indptr, out=None, reduce='add'):
    assert reduce in ['add', 'mean', 'min', 'max']
    assert indptr.dtype == torch.long
    out, arg_out = segment_cuda.segment_csr(src, indptr, out, reduce)
    return out if arg_out is None else (out, arg_out)
