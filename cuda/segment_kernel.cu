#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>

#include "atomics.cuh"
#include "compat.cuh"
#include "index.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

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

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx & (TB - 1);

  if (row_idx < N) {
    auto offset = IndexPtrToOffset<int64_t, int>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = (scalar_t)0;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int src_idx = row_start + lane_idx; src_idx < row_end; src_idx += TB) {
      val += src_data[offset + src_idx];
    }

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2)
      val += __shfl_down_sync(FULL_MASK, val, i); // Parallel reduction

    if (lane_idx == 0) {
      out_data[row_idx] = val;
    }
  }
}

template <typename scalar_t>
__global__ void segment_add_csr_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t *out_data, size_t N, size_t K, size_t E) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    auto offset = IndexPtrToOffset<int64_t, int>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = (scalar_t)0;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int src_idx = row_start; src_idx < row_end; src_idx++) {
      // Coalesced read into `src_data`.
      val += src_data[offset + K * src_idx + lane_idx];
    }

    out_data[thread_idx] = val; // Coalesced write into `out_data`
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
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_csr_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

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

template <typename scalar_t, int TB>
__global__ void segment_add_coo_kernel(const scalar_t *src_data,
                                       const int64_t *index_data,
                                       scalar_t *out_data, size_t numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_idx = thread_idx & (TB - 1);

  if (thread_idx < numel) {
    auto idx = __ldg(index_data + thread_idx);
    scalar_t val = src_data[thread_idx], tmp;

#pragma unroll
    for (int offset = 1; offset < TB; offset *= 2) {
      tmp = __shfl_up_sync(FULL_MASK, val, offset);
      if (lane_idx >= offset &&
          idx == __ldg(index_data + thread_idx - offset)) {
        val += tmp;
      }
    }

    if (lane_idx == TB - 1 || idx != __ldg(index_data + thread_idx + 1)) {
      atomAdd(out_data + idx, val);
    }
  }
}

void segment_add_coo_cuda(at::Tensor src, at::Tensor index, at::Tensor out) {
  auto numel = src.numel();
  auto avg_length = (float)numel / (float)out.numel();

  auto index_data = index.DATA_PTR<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_coo_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    segment_add_coo_kernel<scalar_t, 32>
        <<<BLOCKS(1, numel), THREADS, 0, stream>>>(src_data, index_data,
                                                   out_data, numel);
  });
}

void segment_add_thrust_cuda(at::Tensor src, at::Tensor index, at::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto key = at::full_like(out, -1, out.options().dtype(at::kLong));

  auto index_data = thrust::device_ptr<int64_t>(index.DATA_PTR<int64_t>());
  auto key_data = thrust::device_ptr<int64_t>(key.DATA_PTR<int64_t>());

  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_add_thrust_kernel", [&] {
    auto src_data = thrust::device_ptr<scalar_t>(src.DATA_PTR<scalar_t>());
    auto out_data = thrust::device_ptr<scalar_t>(out.DATA_PTR<scalar_t>());

    thrust::reduce_by_key(policy, index_data, index_data + index.numel(),
                          src_data, key_data, out_data);
  });
}
