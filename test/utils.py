import torch

reductions = ['sum', 'add', 'mean', 'min', 'max']

dtypes = [torch.half, torch.float, torch.double, torch.int, torch.long]
grad_dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, device=device).to(dtype)
