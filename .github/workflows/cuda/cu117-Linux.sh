#!/bin/bash

OS=ubuntu2004

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-${OS}-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo cp /var/cuda-repo-${OS}-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt install cuda-nvcc-11-7 cuda-libraries-dev-11-7
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-${OS}-11-7-local_11.7.1-515.65.01-1_amd64.deb
