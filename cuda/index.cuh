#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

template <typename scalar1, typename scalar2, int64_t Dims>
struct IndexToScatterOffsets3 {
  static __device__ void
  compute(int64_t i, const int64_t dim,
          const at::cuda::detail::TensorInfo<int64_t, int64_t> &index,
          int64_t *indexOffset,
          const at::cuda::detail::TensorInfo<scalar1, int64_t> &t1,
          int64_t *t1Offset,
          const at::cuda::detail::TensorInfo<scalar2, int64_t> &t2,
          int64_t *t2Offset) {
    for (int64_t d = Dims - 1; d >= 0; d--) {
      int64_t curDimIndex = i % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      i /= index.sizes[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    *t2Offset += indexValue * t2.strides[dim];
  }
};

template <typename scalar1, typename scalar2>
struct IndexToScatterOffsets3<scalar1, scalar2, -1> {
  static __device__ void
  compute(int64_t i, const int64_t dim,
          const at::cuda::detail::TensorInfo<int64_t, int64_t> &index,
          int64_t *indexOffset,
          const at::cuda::detail::TensorInfo<scalar1, int64_t> &t1,
          int64_t *t1Offset,
          const at::cuda::detail::TensorInfo<scalar2, int64_t> &t2,
          int64_t *t2Offset) {
    for (int64_t d = index.dims - 1; d >= 0; d--) {
      int64_t curDimIndex = i % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      i /= index.sizes[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    *t2Offset += indexValue * t2.strides[dim];
  }
};

template <typename scalar1, typename scalar2, typename scalar3, int64_t Dims>
struct IndexToScatterOffsets4 {
  static __device__ void
  compute(int64_t i, const int64_t dim,
          const at::cuda::detail::TensorInfo<int64_t, int64_t> &index,
          int64_t *indexOffset,
          const at::cuda::detail::TensorInfo<scalar1, int64_t> &t1,
          int64_t *t1Offset,
          const at::cuda::detail::TensorInfo<scalar2, int64_t> &t2,
          int64_t *t2Offset,
          const at::cuda::detail::TensorInfo<scalar3, int64_t> &t3,
          int64_t *t3Offset) {
    for (int64_t d = Dims - 1; d >= 0; d--) {
      int64_t curDimIndex = i % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
        *t3Offset += curDimIndex * t3.strides[d];
      }
      i /= index.sizes[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    *t2Offset += indexValue * t2.strides[dim];
    *t3Offset += indexValue * t3.strides[dim];
  }
};

template <typename scalar1, typename scalar2, typename scalar3>
struct IndexToScatterOffsets4<scalar1, scalar2, scalar3, -1> {
  static __device__ void
  compute(int64_t i, const int64_t dim,
          const at::cuda::detail::TensorInfo<int64_t, int64_t> &index,
          int64_t *indexOffset,
          const at::cuda::detail::TensorInfo<scalar1, int64_t> &t1,
          int64_t *t1Offset,
          const at::cuda::detail::TensorInfo<scalar2, int64_t> &t2,
          int64_t *t2Offset,
          const at::cuda::detail::TensorInfo<scalar3, int64_t> &t3,
          int64_t *t3Offset) {
    for (int64_t d = index.dims - 1; d >= 0; d--) {
      int64_t curDimIndex = i % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
        *t3Offset += curDimIndex * t3.strides[d];
      }
      i /= index.sizes[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    *t2Offset += indexValue * t2.strides[dim];
    *t3Offset += indexValue * t3.strides[dim];
  }
};
