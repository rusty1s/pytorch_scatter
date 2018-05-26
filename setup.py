from os import path as osp

from setuptools import setup, find_packages

__version__ = '1.0.3'
url = 'https://github.com/rusty1s/pytorch_scatter'

install_requires = ['cffi']
setup_requires = ['pytest-runner', 'cffi']
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
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[osp.join(osp.dirname(__file__), 'build.py:ffi')],
)
