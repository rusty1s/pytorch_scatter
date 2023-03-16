import importlib
import os
import os.path as osp

import torch

__version__ = '2.1.1'

for library in ['_version', '_scatter', '_segment_csr', '_segment_coo']:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    elif os.getenv('BUILD_DOCS', '0') != '1':  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")
    else:  # pragma: no cover
        from .placeholder import cuda_version_placeholder
        torch.ops.torch_scatter.cuda_version = cuda_version_placeholder

        from .placeholder import scatter_placeholder
        torch.ops.torch_scatter.scatter_mul = scatter_placeholder

        from .placeholder import scatter_arg_placeholder
        torch.ops.torch_scatter.scatter_min = scatter_arg_placeholder
        torch.ops.torch_scatter.scatter_max = scatter_arg_placeholder

        from .placeholder import (gather_csr_placeholder,
                                  segment_csr_arg_placeholder,
                                  segment_csr_placeholder)
        torch.ops.torch_scatter.segment_sum_csr = segment_csr_placeholder
        torch.ops.torch_scatter.segment_mean_csr = segment_csr_placeholder
        torch.ops.torch_scatter.segment_min_csr = segment_csr_arg_placeholder
        torch.ops.torch_scatter.segment_max_csr = segment_csr_arg_placeholder
        torch.ops.torch_scatter.gather_csr = gather_csr_placeholder

        from .placeholder import (gather_coo_placeholder,
                                  segment_coo_arg_placeholder,
                                  segment_coo_placeholder)
        torch.ops.torch_scatter.segment_sum_coo = segment_coo_placeholder
        torch.ops.torch_scatter.segment_mean_coo = segment_coo_placeholder
        torch.ops.torch_scatter.segment_min_coo = segment_coo_arg_placeholder
        torch.ops.torch_scatter.segment_max_coo = segment_coo_arg_placeholder
        torch.ops.torch_scatter.gather_coo = gather_coo_placeholder

cuda_version = torch.ops.torch_scatter.cuda_version()
is_not_hip = torch.version.hip is None
is_cuda = torch.version.cuda is not None
if is_not_hip and is_cuda and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_scatter were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_scatter has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_scatter that '
            f'matches your PyTorch install.')

from .scatter import scatter_sum, scatter_add, scatter_mul  # noqa
from .scatter import scatter_mean, scatter_min, scatter_max, scatter  # noqa
from .segment_csr import segment_sum_csr, segment_add_csr  # noqa
from .segment_csr import segment_mean_csr, segment_min_csr  # noqa
from .segment_csr import segment_max_csr, segment_csr, gather_csr  # noqa
from .segment_coo import segment_sum_coo, segment_add_coo  # noqa
from .segment_coo import segment_mean_coo, segment_min_coo  # noqa
from .segment_coo import segment_max_coo, segment_coo, gather_coo  # noqa
from .composite import scatter_std, scatter_logsumexp  # noqa
from .composite import scatter_softmax, scatter_log_softmax  # noqa

__all__ = [
    'scatter_sum',
    'scatter_add',
    'scatter_mul',
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
