#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>

#include "compat.cuh"

std::tuple<at::Tensor, at::Tensor>
segment_add_cuda(at::Tensor src, at::Tensor index, at::Tensor out) {
  cudaSetDevice(src.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto key = at::full_like(out, -1, out.options().dtype(at::kLong));

  auto index_data = thrust::device_ptr<int64_t>(index.DATA_PTR<int64_t>());
  auto key_data = thrust::device_ptr<int64_t>(key.DATA_PTR<int64_t>());

  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_kernel", [&] {
    auto src_data = thrust::device_ptr<scalar_t>(src.DATA_PTR<scalar_t>());
    auto out_data = thrust::device_ptr<scalar_t>(out.DATA_PTR<scalar_t>());

    thrust::reduce_by_key(policy, index_data, index_data + index.size(0),
                          src_data, key_data, out_data);
  });

  return std::make_tuple(out, key);
}
