#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"
#include "compat.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

// We need our own `IndexToOffset` implementation since we do not want to access
// the last element of the `indexptr`.
template <typename T, typename I> struct IndexPtrToOffset {
  static __host__ __device__ I
  get(I idx, const at::cuda::detail::TensorInfo<T, I> &info) {
    I offset = idx % (info.sizes[info.dims - 1] - 1);
    offset *= info.strides[info.dims - 1];
    idx /= info.sizes[info.dims - 1] - 1;
    for (int i = info.dims - 2; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }
    return offset;
  }
};

template <typename scalar_t, int TB>
__global__ void segment_add_csr_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t *out_data, size_t N, size_t E) {

  // Each warp processes exactly `32/TB` rows. We usually set `TB=32` and only
  // make use of it in case the average row length is less than 32.

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx & (TB - 1);

  if (row_idx < N) {
    int offset = IndexPtrToOffset<int64_t, int>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = (scalar_t)0;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int src_idx = row_start + lane_idx; src_idx < row_end; src_idx += TB) {
      val += src_data[offset + src_idx]; // "Mostly" coalesced read.
    }

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2) {
      // Parallel reduction inside a single warp.
      val += __shfl_down_sync(FULL_MASK, val, i);
    }

    if (lane_idx == 0) {
      out_data[row_idx] = val; // "Mostly" coalesced write.
    }
  }
}

template <typename scalar_t>
__global__ void segment_add_csr_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t *out_data, size_t N, size_t K, size_t E) {

  // Each thread processes exactly one row. It turned out that is more efficient
  // than using shared memory due to avoiding synchronization barriers.

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    int offset = IndexPtrToOffset<int64_t, int>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = (scalar_t)0;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int src_idx = row_start; src_idx < row_end; src_idx++) {
      val += src_data[offset + K * src_idx + lane_idx]; // Coalesced read.
    }

    out_data[thread_idx] = val; // Coalesced write.
  }
}

at::Tensor segment_add_csr_cuda(at::Tensor src, at::Tensor indptr) {
  AT_ASSERTM(src.dim() >= indptr.dim());
  for (int i = 0; i < indptr.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == indptr.size(i));

  src = src.contiguous();

  auto reduce_dim = indptr.dim() - 1;
  auto sizes = src.sizes().vec();
  sizes[reduce_dim] = indptr.size(reduce_dim) - 1;
  auto out = at::empty(sizes, src.options());

  auto N = out.size(reduce_dim) * (indptr.numel() / indptr.size(-1));
  auto K = out.numel() / N;
  auto E = src.size(reduce_dim);
  auto avg_length = (float)src.size(reduce_dim) / (float)out.size(reduce_dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_csr_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    // Select the right kernel based on average row length and whether we need
    // broadcasting capabilties (K > 1):
    if (K == 1 && avg_length <= 4) {
      segment_add_csr_kernel<scalar_t, 4><<<BLOCKS(4, N), THREADS, 0, stream>>>(
          src_data, indptr_info, out_data, N, E);
    } else if (K == 1 && avg_length <= 8) {
      segment_add_csr_kernel<scalar_t, 8><<<BLOCKS(8, N), THREADS, 0, stream>>>(
          src_data, indptr_info, out_data, N, E);
    } else if (K == 1 && avg_length <= 16) {
      segment_add_csr_kernel<scalar_t, 16>
          <<<BLOCKS(16, N), THREADS, 0, stream>>>(src_data, indptr_info,
                                                  out_data, N, E);
    } else if (K == 1) {
      segment_add_csr_kernel<scalar_t, 32>
          <<<BLOCKS(32, N), THREADS, 0, stream>>>(src_data, indptr_info,
                                                  out_data, N, E);
    } else {
      segment_add_csr_broadcast_kernel<scalar_t>
          <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(src_data, indptr_info,
                                                     out_data, N, K, E);
    }
  });

  return out;
}

template <typename scalar_t>
__global__ void segment_add_coo_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, size_t E) {

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

#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      tmp = __shfl_up_sync(FULL_MASK, val, i);
      next_idx = __shfl_up_sync(FULL_MASK, idx, i);
      if (lane_idx >= i && idx == next_idx)
        val += tmp;
    }

    next_idx = __shfl_down_sync(FULL_MASK, idx, 1);
    if (lane_idx == 32 - 1 || idx != next_idx) {
      atomAdd(out_data + idx, val);
    }
  }
}

template <typename scalar_t, int TB>
__global__ void segment_add_coo_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, size_t E, size_t K) {

  // Each thread processes a single column and `TB` rows. Coalesced read and
  // write is performed in column-major order. The intermediate results are
  // written via atomics.

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

void segment_add_coo_cuda(at::Tensor src, at::Tensor index, at::Tensor out) {
  AT_ASSERTM(src.dim() >= index.dim());
  for (int i = 0; i < index.dim(); i++)
    AT_ASSERTM(src.size(i) == index.size(i));

  src = src.contiguous();
  auto reduce_dim = index.dim() - 1;

  for (int i = 0; i < out.dim(); i++)
    if (i != reduce_dim)
      AT_ASSERTM(src.size(i) == out.size(i));

  auto E = index.numel();
  auto K = src.numel() / index.numel();
  auto avg_length = (float)src.size(reduce_dim) / (float)out.size(reduce_dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_coo_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    if (K == 1)
      segment_add_coo_kernel<scalar_t><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, E);
    else if (avg_length <= 8)
      segment_add_coo_broadcast_kernel<scalar_t, 4>
          <<<dim3(((E + (8 * 4) - 1) / (8 * 4)), (K + 31) / 32), dim3(32, 8), 0,
             stream>>>(src_data, index_info, out_data, E, K);
    else if (avg_length <= 16)
      segment_add_coo_broadcast_kernel<scalar_t, 8>
          <<<dim3(((E + (8 * 8) - 1) / (8 * 8)), (K + 31) / 32), dim3(32, 8), 0,
             stream>>>(src_data, index_info, out_data, E, K);
    else if (avg_length <= 32)
      segment_add_coo_broadcast_kernel<scalar_t, 16>
          <<<dim3(((E + (8 * 16) - 1) / (8 * 16)), (K + 31) / 32), dim3(32, 8),
             0, stream>>>(src_data, index_info, out_data, E, K);
    else
      segment_add_coo_broadcast_kernel<scalar_t, 32>
          <<<dim3(((E + (8 * 32) - 1) / (8 * 32)), (K + 31) / 32), dim3(32, 8),
             0, stream>>>(src_data, index_info, out_data, E, K);
  });
}
