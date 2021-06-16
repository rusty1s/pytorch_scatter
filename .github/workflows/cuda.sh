#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
echo "1"
sudo apt-get update -qq
echo "2"
sudo apt-cache search cuda-core
echo "3"
sudo apt-get install "cuda-core-11-1" "cuda-nvcc-11-1" "cuda-libraries-dev-11-1"
echo "4"
sudo apt-get install cuda
echo "5"

nvcc --version
