[pypi-image]: https://badge.fury.io/py/torch-scatter.svg
[pypi-url]: https://pypi.python.org/pypi/torch-scatter
[testing-image]: https://github.com/rusty1s/pytorch_scatter/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/rusty1s/pytorch_scatter/actions/workflows/testing.yml
[linting-image]: https://github.com/rusty1s/pytorch_scatter/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/rusty1s/pytorch_scatter/actions/workflows/linting.yml
[docs-image]: https://readthedocs.org/projects/pytorch-scatter/badge/?version=latest
[docs-url]: https://pytorch-scatter.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_scatter/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_scatter?branch=master

# PyTorch Scatter

[![PyPI Version][pypi-image]][pypi-url]
[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]
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

* [**scatter**](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) based on arbitrary indices
* [**segment_coo**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html) based on sorted indices
* [**segment_csr**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html) based on compressed indices via pointers

In addition, we provide the following **composite functions** which make use of `scatter_*` operations under the hood: `scatter_std`, `scatter_logsumexp`, `scatter_softmax` and `scatter_log_softmax`.

All included operations are broadcastable, work on varying data types, are implemented both for CPU and GPU with corresponding backward implementations, and are fully traceable.

## Installation

### Anaconda

**Update:** You can now install `pytorch-scatter` via [Anaconda](https://anaconda.org/rusty1s/pytorch-scatter) for all major OS/PyTorch/CUDA combinations 🤗
Given that you have [`pytorch >= 1.8.0` installed](https://pytorch.org/get-started/locally/), simply run

```
conda install pytorch-scatter -c rusty1s
```

### Binaries

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://pytorch-geometric.com/whl).

#### PyTorch 1.9.0

To install the binaries for PyTorch 1.9.0, simply run

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu111` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

#### PyTorch 1.8.0/1.8.1

To install the binaries for PyTorch 1.8.0 and 1.8.1, simply run

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu101`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu101` | `cu102` | `cu111` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ❌      | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0 and PyTorch 1.7.0/1.7.1 (following the same procedure).

### From source

Ensure that at least PyTorch 1.4.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.4.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
pip install torch-scatter
```

When running in a docker container without NVIDIA driver, PyTorch needs to evaluate the compute capabilities and may fail.
In this case, ensure that the compute capabilities are set via `TORCH_CUDA_ARCH_LIST`, *e.g.*:

```
export TORCH_CUDA_ARCH_LIST = "6.0 6.1 7.2+PTX 7.5+PTX"
```

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

## C++ API

`torch-scatter` also offers a C++ API that contains C++ equivalent of python models.

```
mkdir build
cd build
# Add -DWITH_CUDA=on support for the CUDA if needed
cmake ..
make
make install
```
