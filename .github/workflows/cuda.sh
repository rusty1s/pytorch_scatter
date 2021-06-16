#!/bin/bash

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget -nv https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub

sudo apt-get -qq update

sudo apt install cuda-nvcc-11-1 cuda-libraries-dev-11-1

sudo apt clean

ls -lah /usr/local/

CUDA_HOME=/usr/local/cuda-11.1
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
PATH=${CUDA_HOME}/bin:${PATH}

nvcc --version
