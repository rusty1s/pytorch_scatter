#!/bin/bash

OS=ubuntu1804

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-${OS}-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-${OS}-11-1-local/7fa2af80.pub

sudo apt-get -qq update
sudo apt install cuda-nvcc-11-1 cuda-libraries-dev-11-1
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-${OS}-11-1-local_11.1.1-455.32.00-1_amd64.deb
