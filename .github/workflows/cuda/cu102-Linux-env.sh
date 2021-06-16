#!/bin/bash

CUDA_HOME=/usr/local/cuda-10.2
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
PATH=${CUDA_HOME}/bin:${PATH}
