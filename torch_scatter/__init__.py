# flake8: noqa

import os
import importlib
import os.path as osp

import torch

__version__ = '2.0.3'
expected_torch_version = (1, 4)

try:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        '_version', [osp.dirname(__file__)]).origin)
except OSError as e:
    if 'undefined symbol' in str(e):
        major, minor = [int(x) for x in torch.__version__.split('.')[:2]]
        t_major, t_minor = expected_torch_version
        if major != t_major or (major == t_major and minor != t_minor):
            raise RuntimeError(
                f'Expected PyTorch version {t_major}.{t_minor} but found '
                f'version {major}.{minor}.')
    raise OSError(e)
except AttributeError as e:
    if os.getenv('BUILD_DOCS', '0') == '1':
        pass
    else:
        raise AttributeError(e)

cuda_version = torch.ops.torch_scatter.cuda_version()
if cuda_version != -1 and torch.version.cuda is not None:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]
    cuda_version = str(major) + '.' + str(minor)

    if t_major != major or t_minor != minor:
        raise RuntimeError(
            f'Detected that PyTorch and torch_scatter were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_scatter has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_scatter that '
            f'matches your PyTorch install.')

from .scatter import (scatter_sum, scatter_add, scatter_mean, scatter_min,
                      scatter_max, scatter)
from .segment_csr import (segment_sum_csr, segment_add_csr, segment_mean_csr,
                          segment_min_csr, segment_max_csr, segment_csr,
                          gather_csr)
from .segment_coo import (segment_sum_coo, segment_add_coo, segment_mean_coo,
                          segment_min_coo, segment_max_coo, segment_coo,
                          gather_coo)
from .composite import (scatter_std, scatter_logsumexp, scatter_softmax,
                        scatter_log_softmax)

__all__ = [
    'scatter_sum',
    'scatter_add',
    'scatter_mean',
    'scatter_min',
    'scatter_max',
    'scatter',
    'segment_sum_csr',
    'segment_add_csr',
    'segment_mean_csr',
    'segment_min_csr',
    'segment_max_csr',
    'segment_csr',
    'gather_csr',
    'segment_sum_coo',
    'segment_add_coo',
    'segment_mean_coo',
    'segment_min_coo',
    'segment_max_coo',
    'segment_coo',
    'gather_coo',
    'scatter_std',
    'scatter_logsumexp',
    'scatter_softmax',
    'scatter_log_softmax',
    'torch_scatter',
    '__version__',
]
