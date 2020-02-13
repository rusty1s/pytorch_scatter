#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "$IDX" = "cpu" ]; then
  export TOOLKIT=cpuonly
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "$IDX" = "cu92" ]; then
  export CUDA_SHORT=9.2
  export CUDA=9.2.148-1
  export UBUNTU_VERSION=ubuntu1604
  export CUBLAS=cuda-cublas-dev-9-2
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "$IDX" = "cu100" ]; then
  export CUDA_SHORT=10.0
  export CUDA=10.0.130-1
  export UBUNTU_VERSION=ubuntu1804
  export CUBLAS=cuda-cublas-dev-10-0
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "$IDX" = "cu101" ]; then
  export IDX=cu101
  export CUDA_SHORT=10.1
  export CUDA=10.1.105-1
  export UBUNTU_VERSION=ubuntu1804
  export CUBLAS=libcublas-dev
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "$IDX" = "cpu" ]; then
  export TOOLKIT=cpuonly
fi

if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "$IDX" = "cu92" ]; then
  export CUDA_SHORT=9.2
  export CUDA_URL=https://developer.nvidia.com/compute/cuda/${CUDA_SHORT}/Prod2/local_installers2
  export CUDA_FILE=cuda_${CUDA_SHORT}.148_win10
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "$IDX" = "cu100" ]; then
  export CUDA_SHORT=10.0
  export CUDA_URL=https://developer.nvidia.com/compute/cuda/${CUDA_SHORT}/Prod/local_installers
  export CUDA_FILE=cuda_${CUDA_SHORT}.130_411.31_win10
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "$IDX" = "cu101" ]; then
  export CUDA_SHORT=10.1
  export CUDA_URL=https://developer.nvidia.com/compute/cuda/${CUDA_SHORT}/Prod/local_installers
  export CUDA_FILE=cuda_${CUDA_SHORT}.105_418.96_win10.exe
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${TRAVIS_OS_NAME}" = "osx" ] && [ "$IDX" = "cpu" ]; then
  export TOOLKIT=""
fi

if [ "${IDX}" = "cpu" ]; then
  export FORCE_CPU=1
else
  export FORCE_CUDA=1
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ] && [ "${IDX}" != "cpu" ]; then
  INSTALLER=cuda-repo-${UBUNTU_VERSION}_${CUDA}_amd64.deb
  wget -nv "http://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${INSTALLER}"
  sudo dpkg -i "${INSTALLER}"
  wget -nv "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
  sudo apt-key add 7fa2af80.pub
  sudo apt update -qq
  sudo apt install -y "cuda-core-${CUDA_SHORT/./-}" "cuda-cudart-dev-${CUDA_SHORT/./-}" "${CUBLAS}" "cuda-cusparse-dev-${CUDA_SHORT/./-}"
  sudo apt clean
  CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
  LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
  PATH=${CUDA_HOME}/bin:${PATH}
  nvcc --version
fi

if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "${IDX}" != "cpu" ]; then
  wget -nv "${CUDA_URL}/${CUDA_FILE}"
  PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s nvcc_${CUDA_SHORT} cublas_dev_${CUDA_SHORT} cusparse_dev_${CUDA_SHORT}\" -Wait -NoNewWindow"
  ls /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC
  CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v${CUDA_SHORT}
  PATH=${CUDA_HOME}/bin:$PATH
  PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
  nvcc --version
  cat -n /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v${CUDA_SHORT}/include/crt/host_config.h
fi
