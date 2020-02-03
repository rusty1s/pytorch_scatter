from typing import Optional

import torch
from torch_scatter import scatter_sum
from torch_scatter.utils import broadcast


@torch.jit.script
def scatter_std(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                unbiased: bool = True) -> torch.Tensor:

    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = broadcast(count, tmp, dim).clamp_(1)
    mean = tmp.div_(count)

    var = (src - mean.gather(dim, index))
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count.sub_(1).clamp_(1)
    out.div_(count).sqrt_()

    return out
