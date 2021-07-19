import torch
from torch_scatter import scatter_log_softmax, scatter_softmax


def test_softmax():
    src = torch.tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float('-inf')])
    src.requires_grad_()
    index = torch.tensor([0, 1, 0, 1, 1, 2, 4, 4])

    out = scatter_softmax(src, index)

    out0 = torch.softmax(torch.tensor([0.2, 0.2]), dim=-1)
    out1 = torch.softmax(torch.tensor([0, -2.1, 3.2]), dim=-1)
    out2 = torch.softmax(torch.tensor([7], dtype=torch.float), dim=-1)
    out4 = torch.softmax(torch.tensor([-1, float('-inf')]), dim=-1)

    expected = torch.stack([
        out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]
    ], dim=0)

    assert torch.allclose(out, expected)

    out.backward(torch.randn_like(out))

    jit = torch.jit.script(scatter_softmax)
    assert jit(src, index).tolist() == out.tolist()


def test_log_softmax():
    src = torch.tensor([0.2, 0, 0.2, -2.1, 3.2, 7, -1, float('-inf')])
    src.requires_grad_()
    index = torch.tensor([0, 1, 0, 1, 1, 2, 4, 4])

    out = scatter_log_softmax(src, index)

    out0 = torch.log_softmax(torch.tensor([0.2, 0.2]), dim=-1)
    out1 = torch.log_softmax(torch.tensor([0, -2.1, 3.2]), dim=-1)
    out2 = torch.log_softmax(torch.tensor([7], dtype=torch.float), dim=-1)
    out4 = torch.log_softmax(torch.tensor([-1, float('-inf')]), dim=-1)

    expected = torch.stack([
        out0[0], out1[0], out0[1], out1[1], out1[2], out2[0], out4[0], out4[1]
    ], dim=0)

    assert torch.allclose(out, expected)

    out.backward(torch.randn_like(out))

    jit = torch.jit.script(scatter_log_softmax)
    assert jit(src, index).tolist() == out.tolist()
