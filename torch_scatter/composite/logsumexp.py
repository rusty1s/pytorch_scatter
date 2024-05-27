from typing import Optional

import torch
from torch_scatter import scatter_max, scatter_sum
from torch_scatter.utils import broadcast


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
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
    max_value_per_index = torch.full(
        size,
        fill_value=float('-inf'),
        dtype=src.dtype,
        device=src.device,
    )
    scatter_max(src, index, dim, max_value_per_index, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(torch.isnan(recentered_score), float('-inf'))

    orig_out: Optional[torch.Tensor] = None
    if out is not None:
        orig_out = out.clone()
        out = out.sub_(max_value_per_index).exp_()

    sum_per_index = scatter_sum(recentered_score.exp_(), index, dim, out,
                                dim_size)

    out = sum_per_index.add_(eps).log_().add_(max_value_per_index)

    if orig_out is None:
        return out.nan_to_num_(neginf=0.0)

    mask = ~out.isfinite()
    out[mask] = orig_out[mask]
    return out
