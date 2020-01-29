from typing import Optional

import torch
import torch_scatter


@torch.jit.script
def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None, reduce: str = "sum"):
    return torch.ops.torch_scatter_cpu.segment_sum_csr(src, indptr, out)


def test_jit():
    # op = torch.ops.torch_scatter_cpu.segment_sum_csr

    src = torch.randn(8, 4)
    src.requires_grad_()
    indptr = torch.tensor([0, 2, 4, 6, 8])

    out = segment_csr(src, indptr)
    print(out)

    print(src.grad)
    out.backward(torch.randn_like(out))
    print(src.grad)

    # op = torch.ops.torch_scatter_cpu.segment_csr
    # out = op(src, indptr, None, "sum")
    # print(out)

    # traced_cell = torch.jit.script(op)
