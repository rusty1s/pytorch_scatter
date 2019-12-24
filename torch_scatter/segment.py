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

    # index = index.transpose(dim, -1).contiguous()
    # src = src.transpose(dim, -1).contiguous()
    # out = out.transpose(dim, -1).contiguous()
    # print(index)
    # print(src)

    torch_scatter.segment_cuda.segment_add_thrust(src, index, out)

    # out = out.transpose(dim, -1).contiguous()
    # key = key.transpose(dim, -1).contiguous()

    return out


def segment_add2(src, indptr, dim=-1):
    return torch_scatter.segment_cuda.segment_add(src, indptr, dim)
