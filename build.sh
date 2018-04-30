#!/bin/sh

echo "Compiling kernel..."

if [ -z "$1" ]; then TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))"); else TORCH="$1"; fi
SRC_DIR=torch_scatter/kernel
BUILD_DIR=torch_scatter/build

mkdir -p $BUILD_DIR
$(which nvcc) "-I$TORCH/lib/include" "-I$TORCH/lib/include/TH" "-I$TORCH/lib/include/THC" "-I$SRC_DIR" -c "$SRC_DIR/kernel.cu" -o "$BUILD_DIR/kernel.o" --compiler-options '-fPIC' -std=c++11
