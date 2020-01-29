import torch

torch.ops.load_library('torch_scatter/_C.so')

from .segment_csr import (segment_sum_csr, segment_add_csr, segment_mean_csr,
                          segment_min_csr, segment_max_csr, segment_csr,
                          gather_csr)  # noqa
from .segment_coo import (segment_sum_coo, segment_add_coo, segment_mean_coo,
                          segment_min_coo, segment_max_coo, segment_coo,
                          gather_coo)  # noqa

__version__ = '1.5.0'

__all__ = [
    'segment_sum_csr',
    'segment_add_csr',
    'segment_mean_csr',
    'segment_min_csr',
    'segment_max_csr',
    'segment_max_csr',
    'segment_csr',
    'gather_csr',
    'segment_sum_coo',
    'segment_add_coo',
    'segment_mean_coo',
    'segment_min_coo',
    'segment_max_coo',
    'segment_max_coo',
    'segment_coo',
    'gather_coo',
    'torch_scatter',
    '__version__',
]
