import torch
from torch.testing import get_all_dtypes

dtypes = get_all_dtypes()
dtypes.remove(torch.half)
dtypes.remove(torch.short)  # TODO: PyTorch `atomicAdd` bug with short type.
dtypes.remove(torch.uint8)  # We cannot properly test unsigned values.
dtypes.remove(torch.int8)  # Overflow on gradient computations :(

devices = [torch.device('cpu')]
if torch.cuda.is_available():  # pragma: no cover
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
