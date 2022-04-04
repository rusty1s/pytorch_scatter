#!/bin/bash

OS=ubuntu1804

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda-repo-${OS}-11-5-local_11.5.2-495.29.05-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-11-5-local_11.5.2-495.29.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-${OS}-11-5-local/7fa2af80.pub

sudo apt-get -qq update
sudo apt install cuda-nvcc-11-5 cuda-libraries-dev-11-5
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda-repo-${OS}-11-5-local_11.5.2-495.29.05-1_amd64.deb
