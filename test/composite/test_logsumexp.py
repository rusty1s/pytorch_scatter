import torch
from torch_scatter import scatter_logsumexp


def test_logsumexp():
    src = torch.tensor([0.5, 0, 0.5, -2.1, 3.2, 7, -1, -100])
    index = torch.tensor([0, 1, 0, 1, 1, 2, 4, 4])

    out = scatter_logsumexp(src, index)

    out0 = torch.logsumexp(torch.tensor([0.5, 0.5]), dim=-1)
    out1 = torch.logsumexp(torch.tensor([0, -2.1, 3.2]), dim=-1)
    out2 = torch.logsumexp(torch.tensor(7, dtype=torch.float), dim=-1)
    out3 = torch.logsumexp(torch.tensor([], dtype=torch.float), dim=-1)
    out4 = torch.tensor(-1, dtype=torch.float)

    expected = torch.stack([out0, out1, out2, out3, out4], dim=0)
    assert torch.allclose(out, expected)
