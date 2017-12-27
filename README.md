[build-image]: https://travis-ci.org/rusty1s/pytorch_scatter.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_scatter
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_scatter/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_scatter?branch=master

# PyTorch Scatter [![Build Status][build-image]][build-url] [![Code Coverage][coverage-image]][coverage-url]

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

**[Documentation](http://rusty1s.github.io/pytorch_scatter)**

This package consists of a small extension library of highly optimised sparse update (scatter) operations for the use in [PyTorch](http://pytorch.org/), which are missing in the main package.
Scatter-operations can be roughly described as reduce-operations based on a given "group-index" tensor.
The package consists of the following operations:

* [`scatter_add`](https://rusty2s.github.io/pytorch_scatter/functions/add.html)
* [`scatter_sub`](https://rusty1s.github.io/pytorch_scatter/functions/sub.html)
* [`scatter_mul`](https://rusty1s.github.io/pytorch_scatter/functions/mul.html)
* [`scatter_div`](https://rusty1s.github.io/pytorch_scatter/functions/div.html)
* [`scatter_mean`](https://rusty1s.github.io/pytorch_scatter/functions/mean.html)
* [`scatter_min`](https://rusty1s.github.io/pytorch_scatter/functions/min.html)
* [`scatter_max`](https://rusty1s.github.io/pytorch_scatter/functions/max.html)

All included operations work on varying data types, are implemented both for CPU and GPU and include a backwards implementation.

## Installation

```sh
python setup.py install
```

## Example

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
