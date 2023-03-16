#!/bin/bash

CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8
PATH=${CUDA_HOME}/bin:$PATH
PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="6.0+PTX"
