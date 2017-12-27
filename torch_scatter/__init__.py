from os import path as osp

from .functions.add import scatter_add_, scatter_add
from .functions.sub import scatter_sub_, scatter_sub
from .functions.mul import scatter_mul_, scatter_mul
from .functions.div import scatter_div_, scatter_div
from .functions.mean import scatter_mean_, scatter_mean
from .functions.max import scatter_max_, scatter_max
from .functions.min import scatter_min_, scatter_min

filename = osp.join(osp.dirname(__file__), 'VERSION')
with open(filename, 'r') as f:
    __version__ = f.read().strip()

__all__ = [
    'scatter_add_', 'scatter_add', 'scatter_sub_', 'scatter_sub',
    'scatter_mul_', 'scatter_mul', 'scatter_div_', 'scatter_div',
    'scatter_mean_', 'scatter_mean', 'scatter_max_', 'scatter_max',
    'scatter_min_', 'scatter_min', '__version__'
]
