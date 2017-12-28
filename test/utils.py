import torch
from torch._tensor_docs import tensor_classes

tensors = [t[:-4] for t in tensor_classes]
tensors.remove('ShortTensor')  # TODO: PyTorch `atomicAdd` bug with short type.


def Tensor(str, x):
    tensor = getattr(torch, str)
    return tensor(x)
