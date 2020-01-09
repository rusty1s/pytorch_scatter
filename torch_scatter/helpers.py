import torch


def min_value(dtype):
    try:
        return torch.finfo(dtype).min
    except AttributeError:
        return torch.info(dtype).min


def max_value(dtype):
    try:
        return torch.finfo(dtype).max
    except AttributeError:
        return torch.info(dtype).max
