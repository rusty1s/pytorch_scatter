import torch

from typing import Optional


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def _create_out(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                dim_size: Optional[int] = None,
                is_mul: bool = False) -> torch.Tensor:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    # FIXME: think about whether to fill this with reduction init and use
    # include_self=False or to use torch.empty and include_self=True
    # the former will likely be faster
    # Observation: doing the former for mean will add one to counts which is
    # incorrect
    # Observation: using torch.empty will fill empty places with garbage
    # FIXME: need to use torch.ones for mul
    if is_mul:
        out = torch.ones(size, dtype=src.dtype, device=src.device)
    else:
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out
