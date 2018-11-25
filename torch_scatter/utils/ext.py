import torch
import torch_scatter.scatter_cpu

if torch.cuda.is_available():
    import torch_scatter.scatter_cuda


def get_func(name, tensor):
    if tensor.is_cuda:
        module = torch_scatter.scatter_cuda
    else:
        module = torch_scatter.scatter_cpu
    return getattr(module, name)
