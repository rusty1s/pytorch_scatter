import torch


def min_value(dtype):  # pragma: no cover
    try:
        return torch.finfo(dtype).min
    except TypeError:
        return torch.iinfo(dtype).min


def max_value(dtype):  # pragma: no cover
    try:
        return torch.finfo(dtype).max
    except TypeError:
        return torch.iinfo(dtype).max
