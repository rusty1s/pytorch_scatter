from os import path as osp
from setuptools import setup, find_packages

filename = osp.join(osp.dirname(osp.abspath('__file__')), 'VERSION')
with open(filename, 'r') as f:
    version = f.read().strip()

import build  # noqa

install_requires = ['cffi']
setup_requires = ['pytest-runner', 'cffi']
tests_require = ['pytest', 'pytest-cov']
docs_require = ['Sphinx', 'sphinx_rtd_theme']

setup(
    name='torch_scatter',
    version=version,
    description='PyTorch extension for various scatter methods',
    url='https://github.com/rusty1s/pytorch_scatter',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require + docs_require,
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[osp.join(osp.dirname(__file__), 'build.py:ffi')],
)
