from .._ext import ffi


def get_func(name, tensor):
    name += '_'
    name += 'cuda_' if tensor.is_cuda else ''
    name += tensor.type().split('.')[-1][:-6]
    return getattr(ffi, name)
