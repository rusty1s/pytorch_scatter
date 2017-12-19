from os import path as osp

import torch
from torch.utils.ffi import create_extension

abs_path = osp.join(osp.dirname(osp.realpath(__file__)), 'torch_scatter')
abs_path = 'torch_scatter'

headers = ['torch_scatter/src/cpu.h']
sources = ['torch_scatter/src/cpu.c']
includes = ['torch_scatter/src']
defines = []
extra_objects = []
with_cuda = False

if torch.cuda.is_available():
    headers += ['torch_scatter/src/cuda.h']
    sources += ['torch_scatter/src/cuda.c']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    name='torch_scatter._ext.ffi',
    package=True,
    verbose=True,
    headers=headers,
    sources=sources,
    include_dirs=includes,
    define_macros=defines,
    extra_objects=extra_objects,
    with_cuda=with_cuda,
    relative_to=__file__)

if __name__ == '__main__':
    ffi.build()
