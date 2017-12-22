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

input = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

max, argmax = scatter_max_(index, input, dim=1)

print(max)
 0  0  4  3  2  0
 2  4  3  0  0  0
[torch.FloatTensor of size 2x6]

print(argmax)
-1 -1  3  4  0  1
 1  4  3 -1 -1 -1
[torch.LongTensor of size 2x6]
```
