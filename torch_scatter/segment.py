import torch

from torch_scatter.utils.gen import gen

if torch.cuda.is_available():
    import torch_scatter.segment_cuda


def segment_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    if src.size(dim) == 0:  # pragma: no cover
        return out
    assert src.is_cuda
    torch_scatter.segment_cuda.segment_add(src, index, out, dim)
    return out
