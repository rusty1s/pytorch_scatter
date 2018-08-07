from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [
    CppExtension(
        'scatter_cpu', ['cpu/scatter.cpp'],
        extra_compile_args=['-Wno-unused-variable'])
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('scatter_cuda',
                      ['cuda/scatter.cpp', 'cuda/scatter_kernel.cu'])
    ]

__version__ = '1.0.4'
url = 'https://github.com/rusty1s/pytorch_scatter'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_scatter',
    version=__version__,
    description='PyTorch Extension Library of Optimized Scatter Operations',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'scatter', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
