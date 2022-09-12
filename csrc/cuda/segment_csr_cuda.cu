#include "segment_csr_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "index_info.cuh"
#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

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
        arg_tmp = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
      Reducer<scalar_t, REDUCE>::update(
          &val, SHFL_DOWN_SYNC(FULL_MASK, val, i), &arg, arg_tmp);
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
                 torch::optional<torch::Tensor> optional_out,
                 std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() >= indptr.dim());

  auto sizes = indptr.sizes().vec();
  for (auto i = 0; i < indptr.dim() - 1; i++)
    sizes[i] = src.size(i);
  indptr = indptr.expand(sizes);

  auto dim = indptr.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
    CHECK_INPUT(src.numel() == 0 || out.size(dim) == indptr.size(dim) - 1);
  } else {
    sizes = src.sizes().vec();
    sizes[dim] = std::max<int64_t>(indptr.size(dim) - 1, 0);
    out = torch::empty(sizes, src.options());
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full(out.sizes(), src.size(dim), indptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  auto N = out.size(dim) * (indptr.numel() / indptr.size(-1));
  auto K = out.numel() / N;
  auto E = src.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

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

template <typename scalar_t, int TB>
__global__ void
gather_csr_kernel(const scalar_t *src_data,
                  const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
                  scalar_t *out_data, size_t N, size_t E) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx % TB;

  if (row_idx < N) {
    int offset = IndexPtrToOffset<int64_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = __ldg(src_data + row_idx);

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int out_idx = row_start + lane_idx; out_idx < row_end; out_idx += TB) {
      out_data[offset + out_idx] = val; // "Mostly" coalesced.
    }
  }
}

template <typename scalar_t>
__global__ void gather_csr_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> indptr_info,
    scalar_t *out_data, size_t N, size_t K, size_t E) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    int offset = IndexPtrToOffset<int64_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = src_data[thread_idx]; // Coalesced.

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int out_idx = row_start; out_idx < row_end; out_idx++) {
      out_data[offset + K * out_idx + lane_idx] = val; // "Mostly" coalesced.
    }
  }
}

torch::Tensor gather_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> optional_out) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() >= indptr.dim());

  auto sizes = indptr.sizes().vec();
  for (auto i = 0; i < indptr.dim() - 1; i++)
    sizes[i] = src.size(i);
  indptr = indptr.expand(sizes);

  auto dim = indptr.dim() - 1;
  CHECK_INPUT(src.size(dim) == 0 || src.size(dim) == indptr.size(dim) - 1);

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    auto sizes = src.sizes().vec();
    if (src.numel() > 0) {
      sizes[dim] = indptr.flatten()[-1].cpu().data_ptr<int64_t>()[0];
    } else {
      sizes[dim] = 0;
    }
    out = torch::empty(sizes, src.options());
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return out;
  }

  auto N = src.size(dim) * (indptr.numel() / indptr.size(-1));
  auto K = src.numel() / N;
  auto E = out.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    if (K == 1)
      gather_csr_kernel<scalar_t, 4><<<BLOCKS(1, 4 * N), THREADS, 0, stream>>>(
          src_data, indptr_info, out_data, N, E);
    else
      gather_csr_broadcast_kernel<scalar_t>
          <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(src_data, indptr_info,
                                                     out_data, N, K, E);
  });

  return out;
}
