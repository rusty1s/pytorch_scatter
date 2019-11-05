from itertools import product

import numpy as np
import pytest
import torch
from torch_scatter.composite import scatter_log_softmax, scatter_softmax

from .utils import devices, tensor

SUPPORTED_FLOAT_DTYPES = {torch.float32, torch.float64}


@pytest.mark.parametrize('dtype,device', product(SUPPORTED_FLOAT_DTYPES, devices))
def test_log_softmax(dtype, device):
    src = tensor([0.25, 0, 0.25, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_log_softmax(src, index)

    # Expected results per index
    idx0 = [np.log(0.5), np.log(0.5)]
    idx1 = torch.log_softmax(torch.tensor([0.0, -2.1, 3.2], dtype=dtype), dim=-1).tolist()
    idx2 = 0.0   # Single element, has logprob=0
    # index=3 is empty. Should not matter.
    idx4 = [0.0, float('-inf')]   # log_softmax with -inf preserves the -inf

    np.testing.assert_allclose(
        out.tolist(),
        [idx0[0], idx1[0], idx0[1], idx1[1], idx1[2], idx2, idx4[0], idx4[1]],
        rtol=1e-05, atol=1e-10
        )


@pytest.mark.parametrize('dtype,device', product(SUPPORTED_FLOAT_DTYPES, devices))
def test_softmax(dtype, device):
    src = tensor([0.25, 0, 0.25, -2.1, 3.2, 7, -1, float('-inf')], dtype, device)
    index = tensor([0, 1, 0, 1, 1, 2, 4, 4], torch.long, device)

    out = scatter_softmax(src, index)

    # Expected results per index
    idx0 = [0.5, 0.5]
    idx1 = torch.softmax(torch.tensor([0.0, -2.1, 3.2], dtype=dtype), dim=-1).tolist()
    idx2 = 1   # Single element, has prob=1
    # index=3 is empty. Should not matter.
    idx4 = [1.0, 0.0]   # softmax with -inf yields zero probability

    np.testing.assert_allclose(
        out.tolist(),
        [idx0[0], idx1[0], idx0[1], idx1[1], idx1[2], idx2, idx4[0], idx4[1]],
        rtol=1e-05, atol=1e-10
        )
