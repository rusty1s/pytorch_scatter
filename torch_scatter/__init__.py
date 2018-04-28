from .add import scatter_add
from .sub import scatter_sub
from .mul import scatter_mul
from .div import scatter_div
from .mean import scatter_mean

__version__ = '1.0.0'

__all__ = [
    'scatter_add', 'scatter_sub', 'scatter_mul', 'scatter_div', 'scatter_mean',
    '__version__'
]
