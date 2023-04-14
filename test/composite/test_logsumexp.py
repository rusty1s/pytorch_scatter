import torch
from torch_scatter import scatter_logsumexp


def test_logsumexp():
    inputs = torch.tensor([
        0.5,
        0.5,
        0.0,
        -2.1,
        3.2,
        7.0,
        -1.0,
        -100.0,
    ])
    inputs.requires_grad_()
    index = torch.tensor([0, 0, 1, 1, 1, 2, 4, 4])
    splits = [2, 3, 1, 0, 2]

    outputs = scatter_logsumexp(inputs, index)

    for src, out in zip(inputs.split(splits), outputs.unbind()):
        if src.numel() > 0:
            assert out.tolist() == torch.logsumexp(src, dim=0).tolist()
        else:
            assert out.item() == 0.0

    outputs.backward(torch.randn_like(outputs))

    jit = torch.jit.script(scatter_logsumexp)
    assert jit(inputs, index).tolist() == outputs.tolist()
