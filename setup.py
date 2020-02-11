import os
import os.path as osp
import sys
import glob
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
if os.getenv('FORCE_CUDA', '0') == '1':
    WITH_CUDA = True
if os.getenv('FORCE_NO_CUDA', '0') == '1':
    WITH_CUDA = False

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def get_extensions():
    Extension = CppExtension
    define_macros = []
    extra_compile_args = {'cxx': [], 'nvcc': []}

    # flags = os.getenv('EXTRA_COMPILE_ARGS', '')
    # extra_compile_args['cxx'] += [] if flags == '' else flags.split(' ')
    # extra_compile_args['nvcc'] += [] if flags == '' else flags.split(' ')

    libraries = []

    # Windows users: Make sure that your VS path is included, i.e.:
    # extra_compile_args['cxx'] += ['-I{VISUAL_STUDIO_DIR}\\include']
    # extra_compile_args['nvcc'] += ['-I{VISUAL_STUDIO_DIR}\\include']

    if WITH_CUDA:
        Extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        # extra_compile_args['cxx'] += ['-O0']
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
        extra_compile_args['nvcc'] += nvcc_flags

    if sys.platform == 'win32':
        # extra_compile_args['cxx'] += ['/MP']
        # libraries = ['ATen', '_C']
        pass

    extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    extensions = []
    for main in main_files:
        name = main.split(os.sep)[-1][:-4]

        sources = [main, osp.join(extensions_dir, 'cpu', name + '_cpu.cpp')]
        if WITH_CUDA:
            sources += [osp.join(extensions_dir, 'cuda', name + '_cuda.cu')]

        extension = Extension(
            'torch_scatter._' + name,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
        )
        extensions += [extension]

    return extensions


install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_scatter',
    version='2.0.2',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url='https://github.com/rusty1s/pytorch_scatter',
    description='PyTorch Extension Library of Optimized Scatter Operations',
    keywords=['pytorch', 'scatter', 'segment', 'gather'],
    license='MIT',
    python_requires='>=3.5',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    packages=find_packages(),
)
