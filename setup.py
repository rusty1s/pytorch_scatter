import os
import os.path as osp
import sys
import glob
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
WITH_CUDA = WITH_CUDA or os.getenv('FORCE_CUDA', '0') == '1'


def get_extensions():
    Extension = CppExtension
    define_macros = []
    extra_compile_args = {'cxx': [], 'nvcc': []}

    # Windows users: Edit both of these to contain your VS include path, i.e.:
    # extra_compile_args['cxx'] += ['-I{VISUAL_STUDIO_DIR}\\include']
    # extra_compile_args['nvcc'] += ['-I{VISUAL_STUDIO_DIR}\\include']

    if WITH_CUDA:
        Extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
        extra_compile_args['cxx'] += ['-O0']
        extra_compile_args['nvcc'] += nvcc_flags

    if sys.platform == 'win32':
        extra_compile_args['cxx'] += ['/MP']

    extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    extensions = []
    for main in main_files:
        name = main.split(os.sep)[-1][:-4]

        sources = [main, osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')]
        if WITH_CUDA:
            sources += [osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')]

        extension = Extension(
            f'torch_scatter._{name}',
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
        extensions += [extension]

    return extensions


install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_scatter',
    version='2.0.0',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url='https://github.com/rusty1s/pytorch_scatter',
    description='PyTorch Extension Library of Optimized Scatter Operations',
    keywords=['pytorch', 'scatter', 'segment', 'gather'],
    license='MIT',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    packages=find_packages(),
)
