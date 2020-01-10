import torch


def min_value(dtype):
    try:
        return torch.finfo(dtype).min
    except TypeError:
        return torch.iinfo(dtype).min


def max_value(dtype):
    try:
        return torch.finfo(dtype).max
    except TypeError:
        return torch.iinfo(dtype).max
