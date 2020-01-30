import os.path as osp
from typing import Optional, Tuple

import torch

torch.ops.load_library(
    osp.join(osp.dirname(osp.abspath(__file__)), '_segment_coo.so'))


@torch.jit.script
def segment_sum_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_coo(src, index, out, dim_size)


@torch.jit.script
def segment_add_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_coo(src, index, out, dim_size)


@torch.jit.script
def segment_mean_coo(src: torch.Tensor, index: torch.Tensor,
                     out: Optional[torch.Tensor] = None,
                     dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_coo(src, index, out, dim_size)


@torch.jit.script
def segment_min_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_coo(src, index, out, dim_size)


@torch.jit.script
def segment_max_coo(src: torch.Tensor, index: torch.Tensor,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_coo(src, index, out, dim_size)


@torch.jit.script
def segment_coo(src: torch.Tensor, index: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_coo(src, index, out, dim_size)
    elif reduce == 'mean':
        return segment_mean_coo(src, index, out, dim_size)
    elif reduce == 'min':
        return segment_min_coo(src, index, out, dim_size)[0]
    elif reduce == 'max':
        return segment_max_coo(src, index, out, dim_size)[0]
    else:
        raise ValueError


@torch.jit.script
def gather_coo(src: torch.Tensor, index: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_coo(src, index, out)
