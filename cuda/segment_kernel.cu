#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"
#include "compat.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

enum ReductionType { ADD, MEAN, MIN, MAX };

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    if (reduce == "add") {                                                     \
      const ReductionType REDUCE = ADD;                                        \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "mean") {                                             \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "min") {                                              \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "max") {                                              \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MIN) {
      return std::numeric_limits<scalar_t>::max();
    } else if (REDUCE == MAX) {
      return std::numeric_limits<scalar_t>::min();
    } else {
      return (scalar_t)0;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if ((REDUCE == MIN && new_val < *val) ||
        (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    } else {
      *val = *val + new_val;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == ADD) {
      *address = val;
    } else if (REDUCE == MEAN) {
      *address = val / (scalar_t)max(count, 1);
    } else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else {
        *address = (scalar_t)0;
      }
    }
  }

  static inline __device__ void atom_write(scalar_t *address, scalar_t val,
                                           int64_t *arg_address, int64_t arg) {
    if (REDUCE == ADD) {
      atomAdd(address, val);
    } else if (REDUCE == MEAN) {
      atomAdd(address, val);
    } else if (REDUCE == MIN && val < *address) {
      atomMin(address, val);
    } else if (REDUCE == MAX && val > *address) {
      atomMax(address, val);
    }

    if (REDUCE == MIN || REDUCE == MAX) {
      __syncthreads();
      if (*address == val) {
        *arg_address = arg;
      }
    }
  }
};

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

template <typename scalar_t, ReductionType REDUCE, int TB>
__global__ void
segment_csr_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
                   scalar_t *out_data, int64_t *arg_out_data, size_t N,
                   size_t E) {

  // Each warp processes exactly `32/TB` rows and aggregates all row values
  // via a parallel reduction.

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx & (TB - 1);

  if (row_idx < N) {
    int offset = IndexPtrToOffset<int64_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int64_t arg, arg_tmp;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int src_idx = row_start + lane_idx; src_idx < row_end; src_idx += TB) {
      Reducer<scalar_t, REDUCE>::update(&val, src_data[offset + src_idx], &arg,
                                        src_idx);
    }

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2) {
      // Parallel reduction inside a single warp.
      if (REDUCE == MIN || REDUCE == MAX) {
        arg_tmp = __shfl_down_sync(FULL_MASK, arg, i);
      }
      Reducer<scalar_t, REDUCE>::update(
          &val, __shfl_down_sync(FULL_MASK, val, i), &arg, arg_tmp);
    }

    if (lane_idx == 0) {
      Reducer<scalar_t, REDUCE>::write(out_data + row_idx, val,
                                       arg_out_data + row_idx, arg,
                                       row_end - row_start);
    }
  }
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void segment_csr_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t *out_data, int64_t *arg_out_data, size_t N, size_t K, size_t E) {

  // Each thread processes exactly one row. It turned out that is more
  // efficient than using shared memory due to avoiding synchronization
  // barriers.

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    int offset = IndexPtrToOffset<int64_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int64_t arg;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int src_idx = row_start; src_idx < row_end; src_idx++) {
      Reducer<scalar_t, REDUCE>::update(
          &val, src_data[offset + K * src_idx + lane_idx], &arg, src_idx);
    }

    Reducer<scalar_t, REDUCE>::write(out_data + thread_idx, val,
                                     arg_out_data + thread_idx, arg,
                                     row_end - row_start);
  }
}

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_csr_cuda(at::Tensor src, at::Tensor indptr,
                 at::optional<at::Tensor> out_opt, std::string reduce) {

  AT_ASSERTM(src.dim() >= indptr.dim());
  for (int i = 0; i < indptr.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == indptr.size(i));

  src = src.contiguous();
  auto reduce_dim = indptr.dim() - 1;

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != reduce_dim)
        AT_ASSERTM(src.size(i) == out.size(i));
    AT_ASSERTM(out.size(reduce_dim) == indptr.size(reduce_dim) - 1);
  } else {
    auto sizes = src.sizes().vec();
    sizes[reduce_dim] = indptr.size(reduce_dim) - 1;
    out = at::empty(sizes, src.options());
  }

  at::optional<at::Tensor> arg_out = at::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce == "min" || reduce == "max") {
    arg_out = at::full_like(out, src.size(reduce_dim), indptr.options());
    arg_out_data = arg_out.value().DATA_PTR<int64_t>();
  }

  auto N = out.size(reduce_dim) * (indptr.numel() / indptr.size(-1));
  auto K = out.numel() / N;
  auto E = src.size(reduce_dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_csr_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (K == 1) {
        segment_csr_kernel<scalar_t, REDUCE, 1>
            <<<BLOCKS(32, N), THREADS, 0, stream>>>(
                src_data, indptr_info, out_data, arg_out_data, N, E);
      } else {
        segment_csr_broadcast_kernel<scalar_t, REDUCE>
            <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(
                src_data, indptr_info, out_data, arg_out_data, N, K, E);
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void
segment_coo_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int> index_info,
                   scalar_t *out_data, int64_t *arg_out_data, size_t E) {

  // Each thread processes exactly one entry. Within a warp, we perform a
  // parallel reduction across equal indices, and write the intermediate
  // result via atomics.

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_idx = row_idx & (32 - 1);

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int idx = index_info.data[offset], next_idx;

    scalar_t val = src_data[row_idx], tmp;
    int64_t arg = row_idx % index_info.sizes[index_info.dims - 1], arg_tmp;

#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      // Parallel reduction inside a single warp.
      tmp = __shfl_up_sync(FULL_MASK, val, i);
      if (REDUCE == MIN || REDUCE == MAX) {
        arg_tmp = __shfl_up_sync(FULL_MASK, arg, i);
      }
      next_idx = __shfl_up_sync(FULL_MASK, idx, i);
      assert(idx >= next_idx);
      if (lane_idx >= i && idx == next_idx)
        Reducer<scalar_t, REDUCE>::update(&val, tmp, &arg, arg_tmp);
    }

    next_idx = __shfl_down_sync(FULL_MASK, idx, 1);
    if (lane_idx == 32 - 1 || idx != next_idx) {
      Reducer<scalar_t, REDUCE>::atom_write(out_data + idx, val,
                                            arg_out_data + idx, arg);
    }
  }
}

template <typename scalar_t, ReductionType REDUCE, int TB>
__global__ void segment_coo_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, int64_t *arg_out_data, size_t E, size_t K) {

  // Each thread processes a single column and `TB` index entries. Coalesced
  // read and write is performed in column-major order. The intermediate
  // results are written via atomics.

  int row_start = (blockIdx.x * blockDim.y + threadIdx.y) * TB;
  int col_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (row_start < E && col_idx < K) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_start, index_info);

    int idx1 = __ldg(index_info.data + offset);
    scalar_t val = src_data[K * row_start + col_idx];

#pragma unroll
    for (int i = 1; i < TB; i++) {
      if (row_start + i >= E)
        break;

      int idx2 = __ldg(index_info.data + offset +
                       i * index_info.strides[index_info.dims - 1]);
      assert(idx1 <= idx2);
      if (idx1 == idx2) {
        val += src_data[K * (row_start + i) + col_idx];
      } else {
        atomAdd(out_data + K * idx1 + col_idx, val);
        val = src_data[K * (row_start + i) + col_idx];
      }
      idx1 = idx2;
    }

    atomAdd(out_data + K * idx1 + col_idx, val);
  }
}

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_coo_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                 std::string reduce) {
  AT_ASSERTM(src.dim() >= index.dim());
  for (int i = 0; i < index.dim(); i++)
    AT_ASSERTM(src.size(i) == index.size(i));

  src = src.contiguous();
  out = out.contiguous();
  auto reduce_dim = index.dim() - 1;

  for (int i = 0; i < out.dim(); i++)
    if (i != reduce_dim)
      AT_ASSERTM(src.size(i) == out.size(i));

  at::optional<at::Tensor> arg_out = at::nullopt;
  if (reduce == "min" || reduce == "max") {
    arg_out = at::full_like(out, src.size(reduce_dim), index.options());
  }

  auto E = index.numel();
  auto K = src.numel() / index.numel();
  auto avg_len = (float)src.size(reduce_dim) / (float)out.size(reduce_dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_coo_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    // Select the right kernel based on average row length (purely heuristic)
    // and whether we need broadcasting capabilties (K > 1):

    if (K == 1 && reduce == "add") {
      segment_coo_kernel<scalar_t, ADD><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, nullptr, E);
    } else if (K == 1 && reduce == "mean") {
      segment_coo_kernel<scalar_t, MEAN><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, nullptr, E);
    } else if (K == 1 && reduce == "min") {
      auto arg_out_data = arg_out.value().DATA_PTR<int64_t>();
      segment_coo_kernel<scalar_t, MIN><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, arg_out_data, E);
    } else if (K == 1 && reduce == "max") {
      auto arg_out_data = arg_out.value().DATA_PTR<int64_t>();
      segment_coo_kernel<scalar_t, MAX><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, arg_out_data, E);
    } else if (avg_len <= 8)
      segment_coo_broadcast_kernel<scalar_t, ADD, 4>
          <<<dim3(((E + (8 * 4) - 1) / (8 * 4)), (K + 31) / 32), dim3(32, 8), 0,
             stream>>>(src_data, index_info, out_data, nullptr, E, K);
    else if (avg_len <= 16)
      segment_coo_broadcast_kernel<scalar_t, ADD, 8>
          <<<dim3(((E + (8 * 8) - 1) / (8 * 8)), (K + 31) / 32), dim3(32, 8), 0,
             stream>>>(src_data, index_info, out_data, nullptr, E, K);
    else if (avg_len <= 32)
      segment_coo_broadcast_kernel<scalar_t, ADD, 16>
          <<<dim3(((E + (8 * 16) - 1) / (8 * 16)), (K + 31) / 32), dim3(32, 8),
             0, stream>>>(src_data, index_info, out_data, nullptr, E, K);
    else
      segment_coo_broadcast_kernel<scalar_t, ADD, 32>
          <<<dim3(((E + (8 * 32) - 1) / (8 * 32)), (K + 31) / 32), dim3(32, 8),
             0, stream>>>(src_data, index_info, out_data, nullptr, E, K);
  });

  if (reduce == "mean") {
    AT_ASSERTM(false); // TODO: DIVIDE ENTRIES.
  }

  return std::make_tuple(out, arg_out);
}
