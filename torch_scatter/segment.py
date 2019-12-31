import torch

from torch_scatter.add import scatter_add

if torch.cuda.is_available():
    import torch_scatter.segment_cuda


def segment_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter_add(src, index, dim, out, dim_size, fill_value)


def segment_add_csr(src, indptr, out=None):
    return torch_scatter.segment_cuda.segment_add_csr(src, indptr, out)


def segment_add_coo(src, index, dim_size=None):
    dim_size = index.max().item() + 1 if dim_size is None else dim_size
    size = list(src.size())
    size[index.dim() - 1] = dim_size
    out = src.new_zeros(size)
    torch_scatter.segment_cuda.segment_add_coo(src, index, out)
    return out
