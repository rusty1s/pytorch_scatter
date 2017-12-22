#include <THC.h>

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
__global__ void mulKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0;;
    IndexToScatterOffsets3<Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset);
    atomicMul(&output.data[outputOffset], input.data[inputOffset]);
  }
}

template<typename Real, int Dims>
__global__ void divKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0;;
    IndexToScatterOffsets3<Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset);
    atomicDiv(&output.data[outputOffset], input.data[inputOffset]);
  }
}

template<typename Real, int Dims>
__global__ void meanKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, TensorInfo<Real> count, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0; int countOffset = 0;
    IndexToScatterOffsets4<Real, Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset, count, &countOffset);
    atomicAdd(&output.data[outputOffset], input.data[inputOffset]);
    atomicAdd(&count.data[countOffset], 1);
  }
}

template<typename Real, int Dims>
__global__ void maxKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0;;
    IndexToScatterOffsets3<Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset);
    atomicMax(&output.data[outputOffset], input.data[inputOffset]);
  }
}

template<typename Real, int Dims>
__global__ void minKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0;;
    IndexToScatterOffsets3<Real, Real, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset);
    atomicMin(&output.data[outputOffset], input.data[inputOffset]);
  }
}

template<typename Real, int Dims>
__global__ void argKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> input, TensorInfo<int64_t> arg, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int inputOffset = 0; int argOffset = 0;
    IndexToScatterOffsets4<Real, Real, int64_t, Dims>::compute(i, dim, index, &indexOffset, input, &inputOffset, output, &outputOffset, arg, &argOffset);
    if (eq(input.data[inputOffset], output.data[outputOffset])) arg.data[argOffset] = inputOffset % input.size[dim];
  }
}

template<typename Real, int Dims>
__global__ void indexBackwardKernel(TensorInfo<Real> output, TensorInfo<int64_t> index, TensorInfo<Real> grad, TensorInfo<int64_t> arg, const int dim, const int n) {
  KERNEL_LOOP(i, n) {
    int outputOffset = 0; int indexOffset = 0; int gradOffset = 0; int argOffset = 0;
    IndexToScatterOffsets4<Real, Real, int64_t, Dims>::compute(i, dim, index, &indexOffset, output, &outputOffset, grad, &gradOffset, arg, &argOffset);
    if (arg.data[argOffset] == outputOffset % output.size[dim]) output.data[outputOffset] = grad.data[gradOffset];
  }
}

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
