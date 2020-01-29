#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <torch/extension.h>

#include "atomics.cuh"
#include "compat.cuh"
#include "indptr.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

enum ReductionType { SUM, MEAN, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"add", SUM}, {"mean", MEAN}, {"min", MIN}, {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      const ReductionType REDUCE = SUM;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MIN) {
      return std::numeric_limits<scalar_t>::max();
    } else if (REDUCE == MAX) {
      return std::numeric_limits<scalar_t>::lowest();
    } else {
      return (scalar_t)0;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val,
                                                scalar_t new_val) {
    if (REDUCE == SUM || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if ((REDUCE == MIN && new_val < *val) ||
               (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if ((REDUCE == MIN && new_val < *val) ||
               (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == SUM) {
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

  static inline __device__ void atomic_write(scalar_t *address, scalar_t val) {
    if (REDUCE == SUM || REDUCE == MEAN) {
      atomAdd(address, val);
    } else if (REDUCE == MIN && val < *address) {
      atomMin(address, val);
    } else if (REDUCE == MAX && val > *address) {
      atomMax(address, val);
    }
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
    int64_t row_start = __ldg(indptr_info.data + offset);
    int64_t row_end = __ldg(indptr_info.data + offset +
                            indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int64_t arg, arg_tmp;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int64_t src_idx = row_start + lane_idx; src_idx < row_end;
         src_idx += TB) {
      Reducer<scalar_t, REDUCE>::update(&val, src_data[offset + src_idx], &arg,
                                        src_idx);
    }

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2) {
      // Parallel reduction inside a single warp.
      if (REDUCE == MIN || REDUCE == MAX)
        arg_tmp = __shfl_down_sync(FULL_MASK, arg, i);
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
    int64_t row_start = __ldg(indptr_info.data + offset);
    int64_t row_end = __ldg(indptr_info.data + offset +
                            indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int64_t arg;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int64_t src_idx = row_start; src_idx < row_end; src_idx++) {
      Reducer<scalar_t, REDUCE>::update(
          &val, src_data[offset + K * src_idx + lane_idx], &arg, src_idx);
    }

    Reducer<scalar_t, REDUCE>::write(out_data + thread_idx, val,
                                     arg_out_data + thread_idx, arg,
                                     row_end - row_start);
  }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                 torch::optional<torch::Tensor> out_opt, std::string reduce) {

  cudaSetDevice(src.get_device());

  AT_ASSERTM(src.dim() >= indptr.dim(), "Input mismatch");

  // Broadcasting `indptr` via `expand`.
  auto sizes = indptr.sizes().vec();
  for (int i = 0; i < indptr.dim() - 1; i++) {
    sizes[i] = src.size(i);
  }
  indptr = indptr.expand(sizes);

  src = src.contiguous();
  auto reduce_dim = indptr.dim() - 1;

  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != reduce_dim)
        AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");
    AT_ASSERTM(out.size(reduce_dim) == indptr.size(reduce_dim) - 1,
               "Input mismatch");
  } else {
    sizes = src.sizes().vec();
    sizes[reduce_dim] = indptr.size(reduce_dim) - 1;
    out = torch::empty(sizes, src.options());
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, src.size(reduce_dim), indptr.options());
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

template <typename scalar_t, ReductionType REDUCE, bool HAS_VAL>
__global__ void
segment_coo_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int> index_info,
                   scalar_t *out_data, size_t E, size_t N) {

  // Each thread processes exactly one entry. Within a warp, we perform a
  // parallel reduction across equal indices, and write the intermediate
  // result via atomics.

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_idx = row_idx & (32 - 1);
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset], next_idx;
    int out_idx = (row_idx / D) * N + idx;

    scalar_t val = HAS_VAL ? src_data[row_idx] : (scalar_t)1, tmp;

#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      // Parallel reduction inside a single warp.
      tmp = __shfl_up_sync(FULL_MASK, val, i);
      next_idx = __shfl_up_sync(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        assert(idx >= next_idx);
        if (idx == next_idx)
          Reducer<scalar_t, REDUCE>::update(&val, tmp);
      }
    }

    next_idx = __shfl_down_sync(FULL_MASK, idx, 1);
    if (lane_idx == 32 - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx)
      Reducer<scalar_t, REDUCE>::atomic_write(out_data + out_idx, val);
  }
}

template <typename scalar_t>
__global__ void segment_coo_arg_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, int64_t *arg_out_data, size_t E, size_t N) {

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int64_t idx = index_info.data[offset];
    int out_idx = (row_idx / D) * N + idx;

    scalar_t val = __ldg(out_data + out_idx);
    if (src_data[row_idx] == val)
      arg_out_data[out_idx] = row_idx % D;
  }
}

template <typename scalar_t, ReductionType REDUCE, int TB>
__global__ void segment_coo_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, size_t E, size_t K, size_t N) {

  // Each thread processes a single column and `TB` index entries. Coalesced
  // read and write is performed in column-major order. The intermediate
  // results are written via atomics.

  int D = index_info.sizes[index_info.dims - 1];
  int E_1 = E / D;
  int E_2 = D + TB - (D % TB);

  int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int col_idx = blockIdx.y * blockDim.x + threadIdx.x;

  int dim_start = (row_idx * TB) / E_2;
  int row_start = (row_idx * TB) % E_2;

  if (dim_start < E_1 && col_idx < K) {

    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        dim_start * D + row_start, index_info);
    int idx1 = __ldg(index_info.data + offset), idx2;

    scalar_t val = src_data[K * (dim_start * D + row_start) + col_idx];

#pragma unroll
    for (int i = 1; i < TB; i++) {
      if (row_start + i >= D)
        break;

      idx2 = __ldg(index_info.data + offset +
                   i * index_info.strides[index_info.dims - 1]);
      assert(idx1 <= idx2);
      if (idx1 == idx2) {
        Reducer<scalar_t, REDUCE>::update(
            &val, src_data[K * (dim_start * D + row_start + i) + col_idx]);
      } else {
        Reducer<scalar_t, REDUCE>::atomic_write(
            out_data + (dim_start * N + idx1) * K + col_idx, val);
        val = src_data[K * (dim_start * D + row_start + i) + col_idx];
      }

      idx1 = idx2;
    }

    Reducer<scalar_t, REDUCE>::atomic_write(
        out_data + (dim_start * N + idx1) * K + col_idx, val);
  }
}

template <typename scalar_t>
__global__ void segment_coo_arg_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, int64_t *arg_out_data, size_t E, size_t K, size_t N) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int col_idx = thread_idx % K;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E && col_idx < K) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int idx = __ldg(index_info.data + offset);
    int out_idx = ((row_idx / D) * N + idx) * K + col_idx;

    scalar_t val = __ldg(out_data + out_idx);
    if (src_data[thread_idx] == val)
      arg_out_data[out_idx] = row_idx % D;
  }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 std::string reduce) {

  cudaSetDevice(src.get_device());

  AT_ASSERTM(src.dim() >= index.dim(), "Input mismatch");

  // Broadcasting `index` via `expand`.
  auto sizes = index.sizes().vec();
  for (int i = 0; i < index.dim(); i++) {
    sizes[i] = src.size(i);
  }
  index = index.expand(sizes);

  src = src.contiguous();
  out = out.contiguous();
  auto reduce_dim = index.dim() - 1;

  for (int i = 0; i < out.dim(); i++)
    if (i != reduce_dim)
      AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, src.size(reduce_dim), index.options());
    arg_out_data = arg_out.value().DATA_PTR<int64_t>();
  }

  auto E = index.numel();
  auto E_2 = index.size(reduce_dim);
  auto E_1 = index.numel() / E_2;
  auto K = src.numel() / E;
  auto N = out.size(reduce_dim);
  auto avg_len = (float)E_2 / (float)N;

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_coo_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (K == 1) {
        segment_coo_kernel<scalar_t, REDUCE, true>
            <<<BLOCKS(1, E), THREADS, 0, stream>>>(src_data, index_info,
                                                   out_data, E, N);
      } else if (avg_len <= 8) {
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 4>
            <<<dim3((E_1 * ((E_2 + 3) / 4) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      } else if (avg_len <= 16) {
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 8>
            <<<dim3((E_1 * ((E_2 + 7) / 8) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      } else if (avg_len <= 32) {
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 16>
            <<<dim3((E_1 * ((E_2 + 15) / 16) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      } else {
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 32>
            <<<dim3((E_1 * ((E_2 + 31) / 32) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      }

      if (REDUCE == MIN || REDUCE == MAX) {
        if (K == 1) {
          segment_coo_arg_kernel<scalar_t>
              <<<BLOCKS(1, E), THREADS, 0, stream>>>(
                  src_data, index_info, out_data, arg_out_data, E, N);
        } else {
          segment_coo_arg_broadcast_kernel<scalar_t>
              <<<BLOCKS(1, E * K), THREADS, 0, stream>>>(
                  src_data, index_info, out_data, arg_out_data, E, K, N);
        }
      }
    });
  });

  if (reduce2REDUCE.at(reduce) == MEAN) {
    auto sizes = index.sizes().vec();
    sizes[reduce_dim] = out.size(reduce_dim);
    auto count = torch::zeros(sizes, out.options());

    AT_DISPATCH_ALL_TYPES(out.scalar_type(), "count_kernel", [&] {
      auto count_data = count.DATA_PTR<scalar_t>();
      segment_coo_kernel<scalar_t, SUM, false>
          <<<BLOCKS(1, E), THREADS, 0, stream>>>(nullptr, index_info,
                                                 count_data, E, N);
    });

    count.clamp_(1);
    arg_out = count;

    for (int i = reduce_dim + 1; i < out.dim(); i++) {
      count = count.unsqueeze(-1);
    }

    out.div_(count);
  }

  return std::make_tuple(out, arg_out);
}
