from typing import Optional

import torch
from torch_scatter import scatter_sum, scatter_max

from torch_scatter.utils import broadcast


def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                      out: Optional[torch.Tensor] = None,
                      dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    index = broadcast(index, src, dim)

    if out is not None:
        dim_size = out.size(dim)
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1

    size = list(src.size())
    size[dim] = dim_size

    max_value_per_index = torch.full(size, float('-inf'), dtype=src.dtype,
                                     device=src.device)
    scatter_max(src, index, dim, max_value_per_index, dim_size=dim_size)
    max_value_per_index.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    max_per_src_element = max_value_per_index.gather(dim, index)

    src_recentered = src - max_per_src_element
    if out is not None:
        out = out.sub_(max_value_per_index).exp_()

    sum_per_index = scatter_sum(src_recentered.exp_(), index, dim, out,
                                dim_size)

    return sum_per_index.log_().add_(max_value_per_index)
