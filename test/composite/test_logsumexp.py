from itertools import product

import pytest
import torch
from torch_scatter import scatter_logsumexp
from torch_scatter.testing import float_dtypes, assert_equal

edge_values = [0.0, 1.0, -1e33, 1e33, float("nan"), float("-inf"),
               float("inf")]

tests = [
    [0.5, -2.1, 3.2],
    [],
    *map(list, product(edge_values, edge_values)),
]


@pytest.mark.parametrize('src,dtype', product(tests, float_dtypes))
def test_logsumexp(src, dtype):
    src = torch.tensor(src, dtype=dtype)
    index = torch.zeros_like(src, dtype=torch.long)
    out_scatter = scatter_logsumexp(src, index, dim_size=1)
    out_torch = torch.logsumexp(src, dim=0, keepdim=True)
    assert_equal(out_scatter, out_torch, equal_nan=True)


@pytest.mark.parametrize('src,out', product(tests, edge_values))
def test_logsumexp_inplace(src, out):
    src = torch.tensor(src)
    out = torch.tensor([out])
    out_scatter = out.clone()
    index = torch.zeros_like(src, dtype=torch.long)
    scatter_logsumexp(src, index, out=out_scatter)
    out_torch = torch.logsumexp(torch.cat([out, src]), dim=0, keepdim=True)
    assert_equal(out_scatter, out_torch, equal_nan=True)


def test_logsumexp_parallel_jit():
    splits = [len(src) for src in tests]
    srcs = torch.tensor(sum(tests, start=[]))
    index = torch.repeat_interleave(torch.tensor(splits))

    srcs.requires_grad_()
    outputs = scatter_logsumexp(srcs, index)

    for src, out_scatter in zip(srcs.split(splits), outputs.unbind()):
        out_torch = torch.logsumexp(src, dim=0)
        assert_equal(out_scatter, out_torch, equal_nan=True)

    outputs.backward(torch.randn_like(outputs))

    jit = torch.jit.script(scatter_logsumexp)
    assert_equal(jit(srcs, index), outputs, equal_nan=True)
