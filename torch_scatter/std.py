import torch

from torch_scatter import scatter_add
from torch_scatter.utils.gen import gen


def scatter_std(src, index, dim=-1, out=None, dim_size=None, unbiased=True):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value=0)

    tmp = scatter_add(src, index, dim, None, dim_size)
    count = scatter_add(torch.ones_like(src), index, dim, None, dim_size)
    mean = tmp / count.clamp(min=1)

    var = (src - mean.gather(dim, index))**2
    out = scatter_add(var, index, dim, out, dim_size)
    out = out / (count - 1 if unbiased else count).clamp(min=1)
    out = torch.sqrt(out)

    return out
