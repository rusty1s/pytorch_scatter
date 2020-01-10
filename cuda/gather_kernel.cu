#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "compat.cuh"
#include "indptr.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS

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

at::Tensor gather_csr_cuda(at::Tensor src, at::Tensor indptr,
                           at::optional<at::Tensor> out_opt) {

  AT_ASSERTM(src.dim() >= indptr.dim(), "Input mismatch");
  for (int i = 0; i < indptr.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == indptr.size(i), "Input mismatch");

  src = src.contiguous();
  auto gather_dim = indptr.dim() - 1;
  AT_ASSERTM(src.size(gather_dim) == indptr.size(gather_dim) - 1,
             "Input mismatch");

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != gather_dim)
        AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");
  } else {
    auto d_gather_size = indptr.flatten()[-1].DATA_PTR<int64_t>();
    auto h_gather_size = (int64_t *)malloc(sizeof(int64_t));
    cudaMemcpy(h_gather_size, d_gather_size, sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    auto sizes = src.sizes().vec();
    sizes[gather_dim] = *h_gather_size;
    out = at::empty(sizes, src.options());
  }

  auto N = src.size(gather_dim) * (indptr.numel() / indptr.size(-1));
  auto K = src.numel() / N;
  auto E = out.size(gather_dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int64_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "gather_csr_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    if (K == 1) {
      gather_csr_kernel<scalar_t, 4><<<BLOCKS(1, 4 * N), THREADS, 0, stream>>>(
          src_data, indptr_info, out_data, N, E);
    } else {
      gather_csr_broadcast_kernel<scalar_t>
          <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(src_data, indptr_info,
                                                     out_data, N, K, E);
    }
  });

  return out;
}

template <typename scalar_t>
__global__ void
gather_coo_kernel(const scalar_t *src_data,
                  const at::cuda::detail::TensorInfo<int64_t, int> index_info,
                  scalar_t *out_data, size_t E, size_t N) {

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < E) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int row = index_info.data[offset];

    offset = (row_idx / index_info.sizes[index_info.dims - 1]) * N;
    scalar_t val = __ldg(src_data + offset + row);

    out_data[row_idx] = val;
  }
}

template <typename scalar_t>
__global__ void gather_coo_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out_data, size_t E, size_t K, size_t N) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int col_idx = thread_idx % K;

  if (thread_idx < E * K) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        row_idx, index_info);
    int row = index_info.data[offset];

    offset = (row_idx / index_info.sizes[index_info.dims - 1]) * N * K;
    scalar_t val = __ldg(src_data + offset + K * row + col_idx);

    out_data[thread_idx] = val;
  }
}

at::Tensor gather_coo_cuda(at::Tensor src, at::Tensor index,
                           at::optional<at::Tensor> out_opt) {

  AT_ASSERTM(src.dim() >= index.dim(), "Input mismatch");
  for (int i = 0; i < index.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == index.size(i), "Input mismatch");

  src = src.contiguous();
  auto gather_dim = index.dim() - 1;

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < index.dim(); i++)
      AT_ASSERTM(out.size(i) == index.size(i), "Input mismatch");
    for (int i = index.dim() + 1; i < src.dim(); i++)
      AT_ASSERTM(out.size(i) == src.size(i), "Input mismatch");
  } else {
    auto sizes = src.sizes().vec();
    sizes[gather_dim] = index.size(gather_dim);
    out = at::empty(sizes, src.options());
  }

  auto E = index.numel();
  auto K = out.numel() / E;
  auto N = src.size(gather_dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "gather_coo_kernel", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    if (K == 1) {
      gather_coo_kernel<scalar_t><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, E, N);
    } else {
      gather_coo_broadcast_kernel<scalar_t>
          <<<BLOCKS(1, E * K), THREADS, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
    }
  });

  return out;
}
