#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"
#include "index.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

#define KERNEL_RUN(NAME, DIMS, N, ...)                                         \
  [&] {                                                                        \
    switch (DIMS) {                                                            \
    case 1:                                                                    \
      NAME<scalar_t, 1><<<BLOCKS(N), THREADS>>>(__VA_ARGS__, N);               \
      break;                                                                   \
    case 2:                                                                    \
      NAME<scalar_t, 2><<<BLOCKS(N), THREADS>>>(__VA_ARGS__, N);               \
      break;                                                                   \
    case 3:                                                                    \
      NAME<scalar_t, 3><<<BLOCKS(N), THREADS>>>(__VA_ARGS__, N);               \
      break;                                                                   \
    default:                                                                   \
      NAME<scalar_t, -1><<<BLOCKS(N), THREADS>>>(__VA_ARGS__, N);              \
    }                                                                          \
  }()

template <typename scalar_t, int64_t Dims>
__global__ void
scatter_mul_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> src,
                   at::cuda::detail::TensorInfo<int64_t, int64_t> index,
                   at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                   int64_t dim, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    int64_t srcOffset = 0, indexOffset = 0, outOffset = 0;
    IndexToScatterOffsets3<scalar_t, scalar_t, Dims>::compute(
        i, dim, index, &indexOffset, src, &srcOffset, out, &outOffset);
    atomMul(&out.data[outOffset], src.data[srcOffset]);
  }
}

void scatter_mul_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim) {
  cudaSetDevice(src.get_device());
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_mul_kernel", [&] {
    KERNEL_RUN(scatter_mul_kernel, index.dim(), index.numel(),
               at::cuda::detail::getTensorInfo<scalar_t, int64_t>(src),
               at::cuda::detail::getTensorInfo<int64_t, int64_t>(index),
               at::cuda::detail::getTensorInfo<scalar_t, int64_t>(out), dim);
  });
}

template <typename scalar_t, int64_t Dims>
__global__ void
scatter_div_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> src,
                   at::cuda::detail::TensorInfo<int64_t, int64_t> index,
                   at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                   int64_t dim, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    int64_t srcOffset = 0, indexOffset = 0, outOffset = 0;
    IndexToScatterOffsets3<scalar_t, scalar_t, Dims>::compute(
        i, dim, index, &indexOffset, src, &srcOffset, out, &outOffset);
    atomDiv(&out.data[outOffset], src.data[srcOffset]);
  }
}

void scatter_div_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim) {
  cudaSetDevice(src.get_device());
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_div_kernel", [&] {
    KERNEL_RUN(scatter_div_kernel, index.dim(), index.numel(),
               at::cuda::detail::getTensorInfo<scalar_t, int64_t>(src),
               at::cuda::detail::getTensorInfo<int64_t, int64_t>(index),
               at::cuda::detail::getTensorInfo<scalar_t, int64_t>(out), dim);
  });
}

template <typename scalar_t, int64_t Dims>
__global__ void arg_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> src,
                           at::cuda::detail::TensorInfo<int64_t, int64_t> index,
                           at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                           at::cuda::detail::TensorInfo<int64_t, int64_t> arg,
                           int64_t dim, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    int64_t srcOffset = 0, indexOffset = 0, outOffset = 0, argOffset = 0;
    IndexToScatterOffsets4<scalar_t, scalar_t, int64_t, Dims>::compute(
        i, dim, index, &indexOffset, src, &srcOffset, out, &outOffset, arg,
        &argOffset);
    if (src.data[srcOffset] == out.data[outOffset]) {
      arg.data[argOffset] = (srcOffset / src.strides[dim]) % src.sizes[dim];
    }
  }
}

template <typename scalar_t, int64_t Dims>
__global__ void
scatter_max_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> src,
                   at::cuda::detail::TensorInfo<int64_t, int64_t> index,
                   at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                   int64_t dim, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    int64_t srcOffset = 0, indexOffset = 0, outOffset = 0;
    IndexToScatterOffsets3<scalar_t, scalar_t, Dims>::compute(
        i, dim, index, &indexOffset, src, &srcOffset, out, &outOffset);
    atomMax(&out.data[outOffset], src.data[srcOffset]);
  }
}

void scatter_max_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim) {
  cudaSetDevice(src.get_device());
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_max_kernel", [&] {
    auto src_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(src);
    auto index_info = at::cuda::detail::getTensorInfo<int64_t, int64_t>(index);
    auto out_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(out);
    KERNEL_RUN(scatter_max_kernel, index.dim(), index.numel(), src_info,
               index_info, out_info, dim);
    KERNEL_RUN(arg_kernel, index.dim(), index.numel(), src_info, index_info,
               out_info, at::cuda::detail::getTensorInfo<int64_t, int64_t>(arg),
               dim);
  });
}

template <typename scalar_t, int64_t Dims>
__global__ void
scatter_min_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> src,
                   at::cuda::detail::TensorInfo<int64_t, int64_t> index,
                   at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                   int64_t dim, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    int64_t srcOffset = 0, indexOffset = 0, outOffset = 0;
    IndexToScatterOffsets3<scalar_t, scalar_t, Dims>::compute(
        i, dim, index, &indexOffset, src, &srcOffset, out, &outOffset);
    atomMin(&out.data[outOffset], src.data[srcOffset]);
  }
}

void scatter_min_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim) {
  cudaSetDevice(src.get_device());
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_min_kernel", [&] {
    auto src_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(src);
    auto index_info = at::cuda::detail::getTensorInfo<int64_t, int64_t>(index);
    auto out_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(out);
    KERNEL_RUN(scatter_min_kernel, index.dim(), index.numel(), src_info,
               index_info, out_info, dim);
    KERNEL_RUN(arg_kernel, index.dim(), index.numel(), src_info, index_info,
               out_info, at::cuda::detail::getTensorInfo<int64_t, int64_t>(arg),
               dim);
  });
}
