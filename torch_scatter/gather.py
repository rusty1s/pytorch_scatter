import torch

if torch.cuda.is_available():
    from torch_scatter import gather_cuda


def gather_coo(src, index, out=None):
    return gather_cuda.gather_coo(src, index, out)


def gather_csr(src, indptr, out=None):
    return gather_cuda.gather_csr(src, indptr, out)
