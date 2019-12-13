#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>

#include "compat.cuh"

void segment_add_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim) {
  cudaSetDevice(src.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto index_data = thrust::device_ptr<int64_t>(index.DATA_PTR<int64_t>());
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_kernel", [&] {
    auto src_data = thrust::device_ptr<scalar_t>(src.DATA_PTR<scalar_t>());
    auto out_data = thrust::device_ptr<scalar_t>(out.DATA_PTR<scalar_t>());
  });
}
