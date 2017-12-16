import torch
from torch._tensor_docs import tensor_classes

tensor_strs = [t[:-4] for t in tensor_classes]


def Tensor(str, x):
    tensor = getattr(torch, str)
    return tensor(x)
