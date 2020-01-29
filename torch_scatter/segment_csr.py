from typing import Optional, Tuple

import torch


@torch.jit.script
def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


@torch.jit.script
def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


@torch.jit.script
def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_csr(src, indptr, out)


@torch.jit.script
def segment_min_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_csr(src, indptr, out)


@torch.jit.script
def segment_max_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_csr(src, indptr, out)


@torch.jit.script
def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError


@torch.jit.script
def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_csr(src, indptr, out)
