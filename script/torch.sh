#!/bin/bash

# Fix "member may not be initialized" error on Windows: https://github.com/pytorch/pytorch/issues/27958
if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "${TORCH_VERSION}" != "1.5.0" ]; then
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/runtime/argument_spec.h
  sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
fi

# https://github.com/pytorch/pytorch/commit/d2e16dd888a9b5fd55bd475d4fcffb70f388d4f0
# if [ "${TRAVIS_OS_NAME}" = "windows" ]; then
#   sed -i.bak -e 's/constexpr/CONSTEXPR_EXCEPT_WIN_CUDA/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h
# fi
