template <typename a, typename b, int Dims>
struct IndexToScatterOffsets {
  static __device__ void compute(int i, const int dim,
      const TensorInfo<int64_t>& index, int* indexOffset,
      const TensorInfo<a>& t1, int* t1Offset,
      const TensorInfo<b>& t2, int* t2Offset) {
    int curDimIndex;
    for (int d = Dims - 1; d >= 0; d--) {
      curDimIndex = i % index.size[d];
      *indexOffset += curDimIndex * index.stride[d];
      *t1Offset += curDimIndex * t1.stride[d];
      if (d != dim) *t2Offset += curDimIndex * t2.stride[d];
      i /= index.size[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    assert(indexValue >= 0 && indexValue < t2.size[dim]);
    *t2Offset += indexValue * t2.stride[dim];
  }
};

template <typename a, typename b>
struct IndexToScatterOffsets<a, b, -1> {
  static __device__ void compute(int i, const int dim,
      const TensorInfo<int64_t>& index, int* indexOffset,
      const TensorInfo<a>& t1, int* t1Offset,
      const TensorInfo<b>& t2, int* t2Offset) {
    int curDimIndex;
    for (int d = index.dims - 1; d >= 0; d--) {
      curDimIndex = i % index.size[d];
      *indexOffset += curDimIndex * index.stride[d];
      *t1Offset += curDimIndex * t1.stride[d];
      if (d != dim) *t2Offset += curDimIndex * t2.stride[d];
      i /= index.size[d];
    }
    int64_t indexValue = index.data[*indexOffset];
    assert(indexValue >= 0 && indexValue < t2.size[dim]);
    *t2Offset += indexValue * t2.stride[dim];
  }
};
