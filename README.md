[build-image]: https://travis-ci.org/rusty1s/pytorch_scatter.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_scatter
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_scatter/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_scatter?branch=master

# PyTorch Scatter [![Build Status][build-image]][build-url] [![Code Coverage][coverage-image]][coverage-url]

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

* `scatter_add`
* `scatter_sub`
* `scatter_mul`
* `scatter_div`
* `scatter_mean`
* `scatter_min`
* `scatter_max`

## Installation

```sh
python setup.py install
```

## Usage

```py
from torch_scatter import scatter_max

input = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.LongTensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

max, argmax = scatter_max(index, input, dim=1)
```

```
print(max)
 0  0  4  3  2  0
 2  4  3  0  0  0
[torch.FloatTensor of size 2x6]

print(argmax)
-1 -1  3  4  0  1
 1  4  3 -1 -1 -1
[torch.LongTensor of size 2x6]
```

## Running tests

```sh
python setup.py test
```
