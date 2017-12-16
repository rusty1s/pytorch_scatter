from os import path as osp
from setuptools import setup, find_packages

import build  # noqa

setup(
    name='torch_scatter',
    version='0.1',
    description='PyTorch extension for various scatter methods',
    url='https://github.com/rusty1s/pytorch_scatter',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    install_requires=['cffi>=1.0.0'],
    setup_requires=['cffi>=1.0.0'],
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[osp.join(osp.dirname(__file__), 'build.py:ffi')],
)
