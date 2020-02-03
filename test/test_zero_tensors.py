import torch
from torch_scatter import scatter


def test_zero_elements():
    x = torch.randn(0, 16)
    index = torch.tensor([]).view(0, 16)
    print(x)
    print(index)

    scatter(x, index, dim=0, dim_size=0, reduce="add")
