#!/bin/bash

# Install NVIDIA drivers, see:
# https://github.com/pytorch/vision/blob/master/packaging/windows/internal/cuda_install.bat#L99-L102
curl -k -L "https://drive.google.com/u/0/uc?id=1injUyo3lnarMgWyRcXqKg4UGnN0ysmuq&export=download" --output "/tmp/gpu_driver_dlls.zip"
7z x "/tmp/gpu_driver_dlls.zip" -o"/c/Windows/System32"

export CUDA_SHORT=11.3
export CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
export CUDA_FILE=cuda_${CUDA_SHORT}.0_465.89_win10.exe

# Install CUDA:
curl -k -L "${CUDA_URL}/${CUDA_FILE}" --output "${CUDA_FILE}"
echo ""
echo "Installing from ${CUDA_FILE}..."
PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s nvcc_${CUDA_SHORT} cuobjdump_${CUDA_SHORT} nvprune_${CUDA_SHORT} cupti_${CUDA_SHORT} cublas_dev_${CUDA_SHORT} cudart_${CUDA_SHORT} cufft_dev_${CUDA_SHORT} curand_dev_${CUDA_SHORT} cusolver_dev_${CUDA_SHORT} cusparse_dev_${CUDA_SHORT} thrust_${CUDA_SHORT} npp_dev_${CUDA_SHORT} nvrtc_dev_${CUDA_SHORT} nvml_dev_${CUDA_SHORT}\" -Wait -NoNewWindow"
echo "Done!"
rm -f "${CUDA_FILE}"
