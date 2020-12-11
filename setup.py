import os
import os.path as osp
import glob
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = CUDA_HOME is not None
if os.getenv('FORCE_CUDA', '0') == '1':
    WITH_CUDA = True
if os.getenv('FORCE_CPU', '0') == '1':
    WITH_CUDA = False

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def get_extensions():
    extensions = []
    for with_cuda, supername in [
        (False, "cpu"),
        (True, "gpu"),
    ]:
        if with_cuda and not WITH_CUDA:
            continue
        Extension = CppExtension
        define_macros = []
        extra_compile_args = {'cxx': []}

        if with_cuda:
            Extension = CUDAExtension
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
            extra_compile_args['nvcc'] = nvcc_flags

        extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
        main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
        for main in main_files:
            name = main.split(os.sep)[-1][:-4]

            sources = [main]

            path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
            if osp.exists(path):
                sources += [path]

            path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
            if with_cuda and osp.exists(path):
                sources += [path]

            extension = Extension(
                'torch_scatter._%s_%s' % (name, supername),
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
    version='2.0.5',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url='https://github.com/rusty1s/pytorch_scatter',
    description='PyTorch Extension Library of Optimized Scatter Operations',
    keywords=['pytorch', 'scatter', 'segment', 'gather'],
    license='MIT',
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
)
