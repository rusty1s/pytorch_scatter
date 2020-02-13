#!/bin/bash

if [ "${TRAVIS_OS_NAME}" != "osx" ]; then
  conda install pytorch="${TORCH_VERSION}" "${TOOLKIT}" -c pytorch --yes
fi

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
  conda install pytorch="${TORCH_VERSION}" -c pytorch --yes
fi

# Fix "member may not be initialized" error on Windows: https://github.com/pytorch/pytorch/issues/27958
if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "${IDX}" != "cpu" ]; then
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/script/module.h
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/argument_spec.h
  sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
fi

