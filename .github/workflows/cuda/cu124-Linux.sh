#!/bin/bash

OS=ubuntu2204

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-${OS}-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-${OS}-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt install cuda-nvcc-12-4 cuda-libraries-dev-12-4
sudo apt clean

rm -f https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-${OS}-12-4-local_12.4.1-550.54.15-1_amd64.deb
