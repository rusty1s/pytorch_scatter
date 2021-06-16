#!/bin/bash

export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
