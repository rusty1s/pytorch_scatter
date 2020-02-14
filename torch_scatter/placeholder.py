from typing import Optional, Tuple

import torch


def cuda_version_placeholder() -> int:
    return -1


def scatter_placeholder(src: torch.Tensor, index: torch.Tensor, dim: int,
                        out: Optional[torch.Tensor],
                        dim_size: Optional[int]) -> torch.Tensor:
    raise ImportError
    return src


def scatter_arg_placeholder(src: torch.Tensor, index: torch.Tensor, dim: int,
                            out: Optional[torch.Tensor],
                            dim_size: Optional[int]
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ImportError
    return src, index


def segment_csr_placeholder(src: torch.Tensor, indptr: torch.Tensor,
                            out: Optional[torch.Tensor]) -> torch.Tensor:
    raise ImportError
    return src


def segment_csr_arg_placeholder(src: torch.Tensor, indptr: torch.Tensor,
                                out: Optional[torch.Tensor]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ImportError
    return src, indptr


def gather_csr_placeholder(src: torch.Tensor, indptr: torch.Tensor,
                           out: Optional[torch.Tensor]) -> torch.Tensor:
    raise ImportError
    return src


def segment_coo_placeholder(src: torch.Tensor, index: torch.Tensor,
                            out: Optional[torch.Tensor],
                            dim_size: Optional[int]) -> torch.Tensor:
    raise ImportError
    return src


def segment_coo_arg_placeholder(src: torch.Tensor, index: torch.Tensor,
                                out: Optional[torch.Tensor],
                                dim_size: Optional[int]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ImportError
    return src, index


def gather_coo_placeholder(src: torch.Tensor, index: torch.Tensor,
                           out: Optional[torch.Tensor]) -> torch.Tensor:
    raise ImportError
    return src
