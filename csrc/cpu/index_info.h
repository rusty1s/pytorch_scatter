#pragma once

#include "../extensions.h"

#define MAX_TENSORINFO_DIMS 25

template <typename scalar_t> struct TensorInfo {
  TensorInfo(scalar_t *p, int dim, int sz[MAX_TENSORINFO_DIMS],
             int st[MAX_TENSORINFO_DIMS]) {
    data = p;
    dims = dim;
    AT_ASSERT(dims < MAX_TENSORINFO_DIMS);

    for (int i = 0; i < dim; ++i) {
      sizes[i] = sz[i];
      strides[i] = st[i];
    }
  }

  scalar_t *data;
  int dims;
  int sizes[MAX_TENSORINFO_DIMS];
  int strides[MAX_TENSORINFO_DIMS];
};

template <typename scalar_t>
TensorInfo<scalar_t> getTensorInfo(const torch::Tensor &tensor) {
  int sizes[MAX_TENSORINFO_DIMS];
  int strides[MAX_TENSORINFO_DIMS];

  int dims = tensor.dim();
  for (int i = 0; i < dims; ++i) {
    sizes[i] = tensor.size(i);
    strides[i] = tensor.stride(i);
  }

  return TensorInfo<scalar_t>(tensor.data_ptr<scalar_t>(), dims, sizes,
                              strides);
}

template <typename scalar_t> struct IndexToOffset {
  static inline int get(int idx, const TensorInfo<scalar_t> &info) {
    int offset = 0;
    for (int i = info.dims - 1; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }
    return offset;
  }
};

template <typename scalar_t> struct IndexPtrToOffset {
  static inline int get(int idx, const TensorInfo<scalar_t> &info) {
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
