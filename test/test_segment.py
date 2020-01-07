from itertools import product

import pytest
import torch
from torch_scatter import segment_coo, segment_csr

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_forward(dtype, device):
    # src = tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype,
    #              device)

    src = tensor([1, 2, 3, 4, 5, 6], dtype, device)

    indptr = tensor([0, 2, 5, 5, 6], torch.long, device)
    index = tensor([0, 0, 1, 1, 1, 3], torch.long, device)
    # out = segment_coo(src, index)
    # print('COO', out)

    out = segment_csr(src, indptr, reduce='add')
    print('CSR', out)
    out = segment_csr(src, indptr, reduce='mean')
    print('CSR', out)
    out = segment_csr(src, indptr, reduce='min')
    print('CSR', out)
    out = segment_csr(src, indptr, reduce='max')
    print('CSR', out)


# @pytest.mark.parametrize('dtype,device', product(dtypes, devices))
# def test_benchmark(dtype, device):
#     from torch_geometric.datasets import Planetoid, Reddit  # noqa
#     # data = Planetoid('/tmp/Cora', 'Cora')[0].to(device)
#     data = Planetoid('/tmp/PubMed', 'PubMed')[0].to(device)
#     row, col = data.edge_index
#     print(data.num_edges)
#     print(row.size(0) / data.num_nodes)

#     num_repeats = 1
#     row = row.view(-1, 1).repeat(1, num_repeats).view(-1).contiguous()
#     col = col.view(-1, 1).repeat(1, num_repeats).view(-1).contiguous()

#     # Warmup
#     for _ in range(10):
#         torch.randn(100, 100, device=device).sum()

#     x = torch.randn(row.size(0), device=device)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out1 = scatter_add(x, row, dim=0, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('Scatter Row', time.perf_counter() - t)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         scatter_add(x, col, dim=0, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('Scatter Col', time.perf_counter() - t)

#     rowcount = segment_add(torch.ones_like(row), row)
#     rowptr = torch.cat([rowcount.new_zeros(1), rowcount.cumsum(0)], dim=0)
#     torch.cuda.synchronize()

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out3 = segment_add_csr(x, rowptr)
#     torch.cuda.synchronize()
#     print('CSR', time.perf_counter() - t)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out4 = segment_add_coo(x, row, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('COO', time.perf_counter() - t)

#     assert torch.allclose(out1, out3, atol=1e-2)
#     assert torch.allclose(out1, out4, atol=1e-2)

#     x = torch.randn((row.size(0), 64), device=device)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out5 = scatter_add(x, row, dim=0, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('Scatter Row + Dim', time.perf_counter() - t)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         scatter_add(x, col, dim=0, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('Scatter Col + Dim', time.perf_counter() - t)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out6 = segment_add_csr(x, rowptr)
#     torch.cuda.synchronize()
#     print('CSR + Dim', time.perf_counter() - t)

#     torch.cuda.synchronize()
#     t = time.perf_counter()
#     for _ in range(100):
#         out7 = segment_add_coo(x, row, dim_size=data.num_nodes)
#     torch.cuda.synchronize()
#     print('COO + Dim', time.perf_counter() - t)

#     assert torch.allclose(out5, out6, atol=1e-2)
#     assert torch.allclose(out5, out7, atol=1e-2)
