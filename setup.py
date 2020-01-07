import os.path as osp
from glob import glob
from setuptools import setup, find_packages
from sys import argv

import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

USE_GPU = True
if '--cpu' in argv:
    USE_GPU = False

extra_compile_args = []
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

ext_modules = []
exts = [e.split(osp.sep)[-1][:-4] for e in glob(osp.join('cpu', '*.cpp'))]
ext_modules += [
    CppExtension(f'torch_scatter.{ext}_cpu', [f'cpu/{ext}.cpp'],
                 extra_compile_args=extra_compile_args) for ext in exts
]
# ['-Wno-unused-variable'] if platform.system() != 'Windows' else []

if CUDA_HOME is not None and USE_GPU:
    exts = [e.split(osp.sep)[-1][:-4] for e in glob(osp.join('cuda', '*.cpp'))]
    ext_modules += [
        CUDAExtension(f'torch_scatter.{ext}_cuda',
                      [f'cuda/{ext}.cpp', f'cuda/{ext}_kernel.cu'],
                      extra_compile_args=extra_compile_args) for ext in exts
    ]

__version__ = '1.5.0'
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
    keywords=['pytorch', 'scatter', 'segment'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
