#pragma once

#include <ATen/cuda/detail/TensorInfo.cuh>

// We need our own `IndexToOffset` implementation since we do not want to
// access the last element of the `indexptr`.
template <typename scalar_t> struct IndexPtrToOffset {
  static inline __host__ __device__ int
  get(int idx, const at::cuda::detail::TensorInfo<scalar_t, int> &info) {
    int offset = idx % (info.sizes[info.dims - 1] - 1);
    offset *= info.strides[info.dims - 1];
    idx /= info.sizes[info.dims - 1] - 1;
    for (int i = info.dims - 2; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }
    return offset;
  }
};
