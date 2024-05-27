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


def test_logsumexp_out():
    src = torch.tensor([-1.0, -50.0])
    index = torch.tensor([0, 0])
    out = torch.tensor([-10.0, -10.0])

    scatter_logsumexp(src=src, index=index, out=out)
    assert out.allclose(torch.tensor([-0.9999, -10.0]), atol=1e-4)
