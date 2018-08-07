import torch
import scatter_cpu

if torch.cuda.is_available():
    import scatter_cuda


def get_func(name, tensor):
    module = scatter_cuda if tensor.is_cuda else scatter_cpu
    return getattr(module, name)
