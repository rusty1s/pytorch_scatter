#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
  wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b
  PATH=/home/travis/miniconda3/bin:${PATH}
fi

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
  wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b
  PATH=/Users/travis/miniconda3/bin:${PATH}
fi


if [ "${TRAVIS_OS_NAME}" = "windows" ]; then
  choco install openssl.light
  choco install miniconda3
  PATH=/c/tools/miniconda3/Scripts:$PATH
fi

conda update --yes conda

conda create --yes -n test python="${PYTHON_VERSION}"
source activate test
conda install pytorch="${TORCH_VERSION}" "${TOOLKIT}" -c pytorch --yes

# Fix "member may not be initialized" error on Windows: https://github.com/pytorch/pytorch/issues/27958
if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "${IDX}" != "cpu" ]; then
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/script/module.h
  sed -i.bak -e 's/constexpr/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/argument_spec.h
  sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
fi
