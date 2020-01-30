[pypi-image]: https://badge.fury.io/py/torch-scatter.svg
[pypi-url]: https://pypi.python.org/pypi/torch-scatter
[build-image]: https://travis-ci.org/rusty1s/pytorch_scatter.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_scatter
[docs-image]: https://readthedocs.org/projects/pytorch-scatter/badge/?version=latest
[docs-url]: https://pytorch-scatter.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_scatter/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_scatter?branch=master

# PyTorch Scatter

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

**[Documentation](https://pytorch-scatter.readthedocs.io)**

This package consists of a small extension library of highly optimized sparse update (scatter and segment) operations for the use in [PyTorch](http://pytorch.org/), which are missing in the main package.
Scatter and segment operations can be roughly described as reduce operations based on a given "group-index" tensor.
Segment operations require the "group-index" tensor to be sorted, whereas scatter operations are not subject to these requirements.

The package consists of the following operations with reduction types `"sum"|"mean"|"min"|"max"`:

* [**scatter**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment.html) based on arbitrary indices
* [**segment_coo**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html) based on sorted indices
* [**segment_csr**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html) based on compressed indices via pointers

All included operations are broadcastable, work on varying data types, are implemented both for CPU and GPU with corresponding backward implementations, and are fully traceable.

## Installation

Ensure that at least PyTorch 1.3.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.3.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

When running in a docker container without nvidia driver, PyTorch needs to evaluate the compute capabilities and may fail. In this case, ensure that the compute capabilities are set via `TORCH_CUDA_ARCH_LIST`

```
export TORCH_CUDA_ARCH_LIST = "6.0 6.1 7.2+PTX 7.5+PTX"
```

### Windows

If you are installing this on Windows specifically, **you will need to point the setup to your Visual Studio installation** for some neccessary libraries and header files.
To do this, add the include and library paths of your installation to the path lists in setup.py as described in the respective comments in the code.

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_scatter/issues).
Be sure to import `torch` first before using this package to resolve symbols the dynamic linker must see.

## Example

```py
import torch
from torch_scatter import scatter_max

src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

out, argmax = scatter_max(src, index, dim=-1)
```

```
print(out)
tensor([[0, 0, 4, 3, 2, 0],
        [2, 4, 3, 0, 0, 0]])

print(argmax)
tensor([[5, 5, 3, 4, 0, 1]
        [1, 4, 3, 5, 5, 5]])
```

## Running tests

```
python setup.py test
```
