from itertools import product

import pytest
import torch
import torch_scatter
from torch_scatter.testing import dtypes, reductions, tensor

tests = [
    {
        'src': [1, 2, 3, 4, 5, 6],
        'index': [0, 0, 1, 1, 1, 3],
        'indptr': [0, 2, 5, 5, 6],
        'dim': 0,
        'sum': [3, 12, 0, 6],
        'add': [3, 12, 0, 6],
        'mean': [1.5, 4, 0, 6],
        'min': [1, 3, 0, 6],
        'max': [2, 5, 0, 6],
    },
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='No multiple GPUS')
@pytest.mark.parametrize('test,reduce,dtype', product(tests, reductions,
                                                      dtypes))
def test_forward(test, reduce, dtype):
    device = torch.device('cuda:1')
    src = tensor(test['src'], dtype, device)
    index = tensor(test['index'], torch.long, device)
    indptr = tensor(test['indptr'], torch.long, device)
    dim = test['dim']
    expected = tensor(test[reduce], dtype, device)

    out = torch_scatter.scatter(src, index, dim, reduce=reduce)
    assert torch.all(out == expected)

    out = torch_scatter.segment_coo(src, index, reduce=reduce)
    assert torch.all(out == expected)

    out = torch_scatter.segment_csr(src, indptr, reduce=reduce)
    assert torch.all(out == expected)
