#!/bin/bash

OS=ubuntu2204

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget -nv https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda-repo-${OS}-13-2-local_13.2.0-595.45.04-1_amd64.deb

sudo dpkg -i cuda-repo-${OS}-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo cp /var/cuda-repo-${OS}-13-2-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt install cuda-nvcc-13-2 cuda-libraries-dev-13-2
sudo apt clean

rm -f *.deb
