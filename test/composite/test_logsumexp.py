import torch
from torch_scatter import scatter_logsumexp


def test_logsumexp():
    inputs = torch.tensor([
        0.5, 0.5, 0.0, -2.1, 3.2, 7.0, -1.0, -100.0,
        float('-inf'),
        float('-inf'), 0.0
    ])
    inputs.requires_grad_()
    index = torch.tensor([0, 0, 1, 1, 1, 2, 4, 4, 5, 6, 6])
    splits = [2, 3, 1, 0, 2, 1, 2]

    outputs = scatter_logsumexp(inputs, index)

    for src, out in zip(inputs.split(splits), outputs.unbind()):
        assert out.tolist() == torch.logsumexp(src, dim=0).tolist()

    outputs.backward(torch.randn_like(outputs))
