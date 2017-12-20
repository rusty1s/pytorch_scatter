#include <THC/THC.h>

#include "THCAtomics.cuh"
#include "kernel.h"
#include "common.cuh"

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _kernel_, Real)
#define index_backward TH_CONCAT_2(index_backward_kernel_, Real)
#define check TH_CONCAT_2(check_kernel_, Real)

#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

template <typename Real, int Dims>
__global__ void maxKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, TensorInfo<int64_t> arg, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0; int argOffset = 0;
    int curDimIndex;
    for (int d = index.dims - 1; d >= 0; d--) {
      curDimIndex = i % index.size[d];
      indexOffset += curDimIndex * index.stride[d];
      inputOffset += curDimIndex * input.stride[d];
      if (d != dim) {
        outputOffset += curDimIndex * output.stride[d];
        argOffset += curDimIndex * arg.stride[d];
      }
      i /= index.size[d];
    }
    int64_t indexValue = index.data[indexOffset];
    assert(indexValue >= 0 && indexValue < output.size[dim]);
    outputOffset += indexValue * output.stride[dim];
    argOffset += indexValue * arg.stride[dim];

    atomicMax(&output.data[outputOffset], input.data[inputOffset]);
    // TODO: Do something with arg.
  }
}

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
