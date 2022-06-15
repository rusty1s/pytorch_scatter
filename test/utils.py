import torch

reductions = ['sum', 'add', 'mean', 'min', 'max']

# remove torch.int/long from dtypes as support for int for scatter/segment
# in core is incomplete
dtypes = [torch.half, torch.float, torch.double]
grad_dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, device=device).to(dtype)
