#include "scatter_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const scalar_t *src_data,
               // similarly make IndexType int64 to allow indexing
               // of index with int64s
               const at::cuda::detail::TensorInfo<int64_t, int64_t> index_info,
               scalar_t *out_data,
               // make all the constants int64
               int64_t E, int64_t K, int64_t N, int64_t numel) {

  // changing thread_idx from int32 to int64 so that this logic can handle
  // numel larger than 2^31 - 1
  // FIXME: why is the cast on the RHS necessary?
  // without it the max thread_idx I was getting was 429467296 iirc which is
  // exactly what fits 32 bits, are blockDim.x etc all ints?
  // but online I've never seen code that casts on the RHS
  const int64_t thread_idx = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;

  auto b = thread_idx / (E * K);
  auto k = thread_idx % K;

  if (thread_idx < numel) {
    int64_t offset = at::cuda::detail::IndexToOffset<int64_t, int64_t, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + b * N * K + idx * K + k,
                                            src_data[thread_idx]);
  }
}

template <typename scalar_t>
__global__ void
scatter_arg_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int64_t> index_info,
                   const scalar_t *out_data, int64_t *arg_out_data, int64_t E,
                   int64_t K, int64_t N, int64_t numel) {

  const int64_t thread_idx = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;

  auto b = thread_idx / (E * K);
  auto e = (thread_idx / K) % E;
  auto k = thread_idx % K;

  if (thread_idx < numel) {
    int64_t offset = at::cuda::detail::IndexToOffset<int64_t, int64_t, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    if (src_data[thread_idx] == out_data[b * N * K + idx * K + k]) {
      arg_out_data[b * N * K + idx * K + k] = e;
    }
  }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_cuda(torch::Tensor src, torch::Tensor index, int64_t dim,
             torch::optional<torch::Tensor> optional_out,
             torch::optional<int64_t> dim_size, std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() == index.dim());
  for (auto i = 0; i < index.dim() - 1; i++)
    CHECK_INPUT(src.size(i) >= index.size(i));

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    auto sizes = src.sizes().vec();
    if (dim_size.has_value())
      sizes[dim] = dim_size.value();
    else if (index.numel() == 0)
      sizes[dim] = 0;
    else {
      sizes[dim] = 1 + index.max().cpu().data_ptr<int64_t>()[0];
    }
    out = torch::empty(sizes, src.options());
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, src.size(dim), index.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  auto B = 1;
  for (auto i = 0; i < dim; i++)
    B *= src.size(i);
  auto E = src.size(dim);
  auto K = src.numel() / (B * E);
  auto N = out.size(dim);

  // make IndexType int64 to allow indexing of index with int64s
  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int64_t>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (!optional_out.has_value())
        out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<scalar_t, REDUCE>
          <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
              src_data, index_info, out_data, E, K, N, src.numel());

      if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);

      // Comment arg kernel out so comparison for benchmarks is fair
      // if (REDUCE == MIN || REDUCE == MAX)
      //   scatter_arg_kernel<scalar_t>
      //       <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
      //           src_data, index_info, out_data, arg_out_data, E, K, N,
      //           src.numel());
    });
  });

  return std::make_tuple(out, arg_out);
}
