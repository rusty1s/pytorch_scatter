from os import path as osp

from torch.utils.ffi import create_extension

abs_path = osp.join(osp.dirname(osp.realpath(__file__)), 'torch_scatter')

headers = ['torch_scatter/include/scatter.h']
sources = ['torch_scatter/src/scatter.c']
includes = [osp.join(abs_path, 'include'), osp.join(abs_path, 'src')]
defines = []
extra_objects = []
with_cuda = False

ffi = create_extension(
    name='torch_scatter._ext.scatter',
    package=True,
    verbose=True,
    headers=headers,
    sources=sources,
    includes=includes,
    define_macros=defines,
    extra_objects=extra_objects,
    with_cuda=with_cuda,
    relative_to=__file__)

if __name__ == '__main__':
    ffi.build()
