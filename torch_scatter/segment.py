import torch

from torch_scatter.utils.gen import gen
from torch_scatter.add import scatter_add

if torch.cuda.is_available():
    import torch_scatter.segment_cuda


def segment_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    if src.size(dim) == 0:  # pragma: no cover
        return out

    if not src.is_cuda:
        return scatter_add(src, index, dim, out, dim_size, fill_value)

    torch_scatter.segment_cuda.segment_add_thrust(src, index, out)

    return out


def segment_add_csr(src, indptr):
    return torch_scatter.segment_cuda.segment_add_csr(src, indptr)


def segment_add_coo(src, index, dim_size=None):
    dim_size = index.max().item() + 1 if dim_size is None else dim_size
    out = src.new_zeros(dim_size)
    torch_scatter.segment_cuda.segment_add_coo(src, index, out)
    return out
