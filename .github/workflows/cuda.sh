#!/bin/bash

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb

# sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb

# sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub

# sudo apt-get update

# sudo apt-get -y install cuda

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update -qq
sudo apt-cache search cuda-core
sudo apt-get install "cuda-core-11-1" "cuda-nvcc-11-1" "cuda-libraries-dev-11-1"

nvcc --version



# export UBUNTU_VERSION=ubuntu2004
# export CUDA=11-

# INSTALLER="cuda-repo-${UBUNTU_VERSION}_${CUDA}_amd64.deb"
# wget -nv "http://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${INSTALLER}"
# sudo dpkg -i "${INSTALLER}"
# wget -nv "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
# sudo apt-key add 7fa2af80.pub
# sudo apt update -qq
# sudo apt install "cuda-core-${CUDA_SHORT/./-}" "cuda-nvcc-${CUDA_SHORT/./-}" "cuda-libraries-dev-${CUDA_SHORT/./-}"
# sudo apt clean
# CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
# LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# PATH=${CUDA_HOME}/bin:${PATH}
# nvcc --version

