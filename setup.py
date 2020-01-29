import platform
import os.path as osp
from glob import glob
from setuptools import setup, find_packages
from sys import argv

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

# Windows users: Edit both of these to contain your VS include path, i.e.:
# cxx_extra_compile_args = ['-I{VISUAL_STUDIO_DIR}\\include']
# nvcc_extra_compile_args = [..., '-I{VISUAL_STUDIO_DIR}\\include']
cxx_extra_compile_args = []
nvcc_extra_compile_args = ['-arch=sm_35', '--expt-relaxed-constexpr']

# Windows users: Edit both of these to contain your VS library path, i.e.:
# cxx_extra_link_args = ['/LIBPATH:{VISUAL_STUDIO_DIR}\\lib\\{x86|x64}']
# nvcc_extra_link_args = ['/LIBPATH:{VISUAL_STUDIO_DIR}\\lib\\{x86|x64}']
cxx_extra_link_args = []
nvcc_extra_link_args = []

if platform.system() != 'Windows':
    cxx_extra_compile_args += ['-Wno-unused-variable']
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    cxx_extra_compile_args += ['-DVERSION_GE_1_3']
    nvcc_extra_compile_args += ['-DVERSION_GE_1_3']
cmdclass = {
    'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
}

ext_modules = []
exts = [e.split(osp.sep)[-1][:-4] for e in glob(osp.join('cpu', '*.cpp'))]
ext_modules += [
    CppExtension(f'torch_scatter.{ext}_cpu', [f'cpu/{ext}.cpp'],
                 extra_compile_args=cxx_extra_compile_args,
                 extra_link_args=cxx_extra_link_args) for ext in exts
]

if CUDA_HOME is not None and '--cpu' not in argv:
    exts = [e.split(osp.sep)[-1][:-4] for e in glob(osp.join('cuda', '*.cpp'))]
    ext_modules += [
        CUDAExtension(
            f'torch_scatter.{ext}_cuda',
            [f'cuda/{ext}.cpp', f'cuda/{ext}_kernel.cu'], extra_compile_args={
                'cxx': cxx_extra_compile_args,
                'nvcc': nvcc_extra_compile_args,
            }, extra_link_args=nvcc_extra_link_args) for ext in exts
    ]
if '--cpu' in argv:
    argv.remove('--cpu')

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
