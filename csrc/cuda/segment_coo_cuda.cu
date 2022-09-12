#include "segment_coo_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

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
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        assert(idx >= next_idx);
        if (idx == next_idx)
          Reducer<scalar_t, REDUCE>::update(&val, tmp);
      }
    }

    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
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
  int E_2 = (D - 1) + TB - ((D - 1) % TB);

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
segment_coo_cuda(torch::Tensor src, torch::Tensor index,
                 torch::optional<torch::Tensor> optional_out,
                 torch::optional<int64_t> dim_size, std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() >= index.dim());

  auto sizes = index.sizes().vec();
  for (int i = 0; i < index.dim(); i++) {
    sizes[i] = src.size(i);
  }
  index = index.expand(sizes);

  auto dim = index.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    sizes = src.sizes().vec();
    if (dim_size.has_value())
      sizes[dim] = dim_size.value();
    else if (index.numel() == 0)
      sizes[dim] = 0;
    else {
      auto tmp = index.select(dim, index.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      sizes[dim] = 1 + tmp.cpu().data_ptr<int64_t>()[0];
    }
    out = torch::zeros(sizes, src.options());
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, src.size(dim), index.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  } else if (reduce2REDUCE.at(reduce) == MEAN) {
    auto sizes = index.sizes().vec();
    sizes[dim] = out.size(dim);
    arg_out = torch::zeros(sizes, out.options());
  }

  if (index.numel() == 0)
    return std::make_tuple(out, arg_out);

  auto E = index.numel();
  auto E_2 = index.size(dim);
  auto E_1 = index.numel() / E_2;
  auto K = src.numel() / E;
  auto N = out.size(dim);
  auto avg_len = (float)E_2 / (float)N;

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (!optional_out.has_value())
        out.fill_(Reducer<scalar_t, REDUCE>::init());

      if (K == 1)
        segment_coo_kernel<scalar_t, REDUCE, true>
            <<<BLOCKS(1, E), THREADS, 0, stream>>>(src_data, index_info,
                                                   out_data, E, N);
      else if (avg_len <= 8)
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 4>
            <<<dim3((E_1 * ((E_2 + 3) / 4) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      else if (avg_len <= 16)
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 8>
            <<<dim3((E_1 * ((E_2 + 7) / 8) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      else if (avg_len <= 32)
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 16>
            <<<dim3((E_1 * ((E_2 + 15) / 16) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);
      else
        segment_coo_broadcast_kernel<scalar_t, REDUCE, 32>
            <<<dim3((E_1 * ((E_2 + 31) / 32) + 7) / 8, (K + 31) / 32),
               dim3(32, 8), 0, stream>>>(src_data, index_info, out_data, E, K,
                                         N);

      if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);

      if (REDUCE == MIN || REDUCE == MAX) {
        if (K == 1)
          segment_coo_arg_kernel<scalar_t>
              <<<BLOCKS(1, E), THREADS, 0, stream>>>(
                  src_data, index_info, out_data, arg_out_data, E, N);
        else
          segment_coo_arg_broadcast_kernel<scalar_t>
              <<<BLOCKS(1, E * K), THREADS, 0, stream>>>(
                  src_data, index_info, out_data, arg_out_data, E, K, N);
      }

      if (REDUCE == MEAN) {
        auto count_data = arg_out.value().data_ptr<scalar_t>();
        segment_coo_kernel<scalar_t, SUM, false>
            <<<BLOCKS(1, E), THREADS, 0, stream>>>(nullptr, index_info,
                                                   count_data, E, N);
        arg_out.value().masked_fill_(arg_out.value() < (scalar_t)1,
                                     (scalar_t)1);
        auto count = arg_out.value();
        for (int i = dim + 1; i < out.dim(); i++)
          count = count.unsqueeze(-1);
        if (out.is_floating_point())
          out.true_divide_(count);
        else
          out.div_(count, "floor");
      }
    });
  });

  return std::make_tuple(out, arg_out);
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

torch::Tensor gather_coo_cuda(torch::Tensor src, torch::Tensor index,
                              torch::optional<torch::Tensor> optional_out) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() >= index.dim());

  auto sizes = index.sizes().vec();
  for (auto i = 0; i < index.dim() - 1; i++)
    sizes[i] = src.size(i);
  index = index.expand(sizes);

  auto dim = index.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < src.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
    CHECK_INPUT(index.size(dim) == out.size(dim));
  } else {
    auto sizes = src.sizes().vec();
    sizes[dim] = index.size(dim);
    out = torch::empty(sizes, src.options());
  }

  if (index.numel() == 0)
    return out;

  auto E = index.numel();
  auto K = out.numel() / E;
  auto N = src.size(dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    if (K == 1)
      gather_coo_kernel<scalar_t><<<BLOCKS(1, E), THREADS, 0, stream>>>(
          src_data, index_info, out_data, E, N);
    else
      gather_coo_broadcast_kernel<scalar_t>
          <<<BLOCKS(1, E * K), THREADS, 0, stream>>>(src_data, index_info,
                                                     out_data, E, K, N);
  });

  return out;
}
