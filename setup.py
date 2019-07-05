import platform
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

extra_compile_args = []
if platform.system() != 'Windows':
    extra_compile_args += ['-Wno-unused-variable']

ext_modules = [
    CppExtension(
        'torch_scatter.scatter_cpu', ['cpu/scatter.cpp'],
        extra_compile_args=extra_compile_args)
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_scatter.scatter_cuda',
                      ['cuda/scatter.cpp', 'cuda/scatter_kernel.cu'])
    ]

__version__ = '1.3.1'
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
    keywords=[
        'pytorch',
        'scatter',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
