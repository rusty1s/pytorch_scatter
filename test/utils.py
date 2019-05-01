import torch
from torch.testing import get_all_dtypes

dtypes = get_all_dtypes()

dtypes.remove(torch.half)
dtypes.remove(torch.short)  # PyTorch scatter does not work on short types.
dtypes.remove(torch.bool)

grad_dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
