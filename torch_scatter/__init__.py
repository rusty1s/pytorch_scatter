import torch

from .add import scatter_add
from .sub import scatter_sub
from .mul import scatter_mul
from .div import scatter_div
from .mean import scatter_mean
from .std import scatter_std
from .max import scatter_max
from .min import scatter_min
from .logsumexp import scatter_logsumexp

from .segment import segment_coo, segment_csr
from .gather import gather_coo, gather_csr

import torch_scatter.composite

torch.ops.load_library('torch_scatter/scatter_cpu.so')
torch.ops.load_library('torch_scatter/segment_cpu.so')
torch.ops.load_library('torch_scatter/gather_cpu.so')

try:
    torch.ops.load_library('torch_scatter/scatter_cuda.so')
    torch.ops.load_library('torch_scatter/segment_cuda.so')
    torch.ops.load_library('torch_scatter/gather_cuda.so')
except OSError as e:
    if torch.cuda.is_available():
        raise e

__version__ = '1.4.0'

__all__ = [
    'scatter_add',
    'scatter_sub',
    'scatter_mul',
    'scatter_div',
    'scatter_mean',
    'scatter_std',
    'scatter_max',
    'scatter_min',
    'scatter_logsumexp',
    'segment_coo',
    'segment_csr',
    'gather_coo',
    'gather_csr',
    'torch_scatter',
    '__version__',
]
