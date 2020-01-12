from __future__ import division

from itertools import repeat

import torch


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    dim = index.max().item() + 1 if index.numel() > 0 else 0
    return int(dim)


def broadcast(src, index, dim):
    dim = range(src.dim())[dim]  # Get real dim value.

    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        if index.numel() > 0:
            index = index.view(index_size).expand_as(src)
        else:  # pragma: no cover
            # PyTorch has a bug when view is used on zero-element tensors.
            index = src.new_empty(index_size, dtype=torch.long)

    # Broadcasting capabilties: Expand dimensions to match.
    if src.dim() != index.dim():
        raise ValueError(
            ('Number of dimensions of src and index tensor do not match, '
             'got {} and {}').format(src.dim(), index.dim()))

    expand_size = []
    for s, i in zip(src.size(), index.size()):
        expand_size += [-1 if s == i and s != 1 and i != 1 else max(i, s)]

    src = src.expand(expand_size)
    index = index.expand_as(src)

    return src, index


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, index = broadcast(src, index, dim)
    dim = range(src.dim())[dim]  # Get real dim value.

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim
