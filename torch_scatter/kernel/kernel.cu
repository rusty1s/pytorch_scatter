#include <THC/THC.h>

#include "kernel.h"

#include "common.cuh"
#include "THCIndex.cuh"
#include "THCAtomics.cuh"

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _kernel_, Real)
#define index_backward TH_CONCAT_2(index_backward_kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

template<typename Real, int Dims>
__global__ void maxKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, TensorInfo<int64_t> arg, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0;
    IndexToScatterOffsets<Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset);
    atomicMax(&output.data[outputOffset], input.data[inputOffset]);
    // TODO: Do something with arg.
  }
}

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
