from typing import Any

import torch

reductions = ['sum', 'add', 'mean', 'min', 'max']

dtypes = [
    torch.half, torch.bfloat16, torch.float, torch.double, torch.int,
    torch.long
]
grad_dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:0')]


def tensor(x: Any, dtype: torch.dtype, device: torch.device):
    return None if x is None else torch.tensor(x, device=device).to(dtype)
