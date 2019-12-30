import time
from itertools import product

import pytest
import torch
from torch_scatter import segment_add, scatter_add
from torch_scatter.segment import segment_add_csr, segment_add_coo

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward(dtype, device):
    src = tensor([1, 2, 3, 4, 5, 6], dtype, device)
    index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)
    out = segment_add(src, index, dim=0)
    # print('Thrust', out)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward2(dtype, device):
    src = tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype,
                 device)
    indptr = tensor([0, 2, 5, 5, 6], torch.long, device)
    # indptr = indptr.view(1, -1).expand(2, -1).t().contiguous().t()

    out = segment_add_csr(src, indptr)
    print('CSR', out)

    # index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)
    # out = segment_add_coo(src, index)
    # print('COO', out)


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_benchmark(dtype, device):
    from torch_geometric.datasets import Planetoid, Reddit  # noqa
    # data = Planetoid('/tmp/Cora', 'Cora')[0].to(device)
    data = Planetoid('/tmp/PubMed', 'PubMed')[0].to(device)
    # data = Reddit('/tmp/Reddit')[0].to(device)
    row, col = data.edge_index
    x = torch.randn(data.num_edges, device=device)
    print(row.size(0) / data.num_nodes)

    # Warmup
    for _ in range(10):
        torch.randn(100, 100, device=device).sum()

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out1 = scatter_add(x, row, dim=0, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('Scatter Row', time.perf_counter() - t)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        scatter_add(x, col, dim=0, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('Scatter Col', time.perf_counter() - t)

    torch.cuda.synchronize()

    t = time.perf_counter()
    for _ in range(100):
        out2 = segment_add(x, row, dim=0, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('Thrust', time.perf_counter() - t)

    assert torch.allclose(out1, out2, atol=1e-2)

    rowcount = segment_add(torch.ones_like(row), row)
    rowptr = torch.cat([rowcount.new_zeros(1), rowcount.cumsum(0)], dim=0)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out3 = segment_add_csr(x, rowptr)
    torch.cuda.synchronize()
    print('CSR', time.perf_counter() - t)

    assert torch.allclose(out1, out3, atol=1e-2)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out4 = segment_add_coo(x, row, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('COO', time.perf_counter() - t)

    assert torch.allclose(out1, out4, atol=1e-2)

    x = torch.randn((data.num_edges, 32), device=device)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out5 = scatter_add(x, row, dim=0, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('Scatter Row + Dim', time.perf_counter() - t)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        scatter_add(x, col, dim=0, dim_size=data.num_nodes)
    torch.cuda.synchronize()
    print('Scatter Col + Dim', time.perf_counter() - t)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        out6 = segment_add_csr(x, rowptr)
    torch.cuda.synchronize()
    print('CSR + Dim', time.perf_counter() - t)

    assert torch.allclose(out5, out6, atol=1e-2)
