# PyTorch Scatter

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

```sh
pip install torch_scatter
```

```py
from torch_scatter import scatter_max

input = torch.Tensor([[2, 0, 1, 4, 3], [0,2, 1, 3, 4]])
torch.LongTensor([[4, 5, 2, 3], [0, 0, 2, 2, 1]])

max, arg = scatter_max_(index, input, dim=1)
```

## Features

* CPU and Cuda implementation
* Backward implementation
