#include "ATen/ATen.h"
#include "ATen/Parallel.h"

#include "scatter_cpu.h"

#include "index_info.h"
#include "reducer.h"
#include "utils.h"

#include <stdio.h>
#include <mutex>

const int N_MUTEXES = 256;
const int B_GRAIN_SIZE = 512;
const int E_GRAIN_SIZE = 512;
std::array<std::mutex, N_MUTEXES> mutexes;

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_cpu(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size, std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

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
    else
      sizes[dim] = 1 + *index.max().data_ptr<int64_t>();
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

  // one mutex per K * N / N_MUTEXES indices
  //  - mutex i owns [i * gap, i + gap]
  //  - index j gets mutex (j / gap)
  int gap = std::max((long)ceil((K * N) / (float)N_MUTEXES), 1L);
  auto index_info = getTensorInfo<int64_t>(index);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (!optional_out.has_value())
        out.fill_(Reducer<scalar_t, REDUCE>::init());

      /* 
      Design notes
      ------------
      - If one only parallelizes over the batch dimension (e.g. dimensions
        before `dim`), no mutexes are required as each thread will write to a 
        unique value in `out`).
      - If one parallelizes over any other dimension, mutexes are required to
        ensure non-corrupted writes.
      */

      // batch dimension(s) [parallel]
      at::parallel_for(0, B, B_GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int b = begin; b < end; b++) {

          // scatter dimension [parallel]
          at::parallel_for(0, E, E_GRAIN_SIZE, [&](int64_t begin_2, int64_t end_2) { 
            for(auto e = begin_2; e < end_2; e++) {
              
              // all other dimensions [serial]
              for(int k = 0; k < K; k++) {
                int64_t i = b * E * K + e * K + k;
                int64_t idx = index_info.data[IndexToOffset<int64_t>::get(i, index_info)];

                int out_idx = b * N * K + idx * K + k;
                int mtx_idx = (idx * K + k) / gap;
                mutexes[mtx_idx].lock();
                Reducer<scalar_t, REDUCE>::update(
                    out_data + out_idx, src_data[i],
                    arg_out_data + out_idx, e);
                mutexes[mtx_idx].unlock();
              }
            }
          });
        }
      });

      if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
    });
  });

  return std::make_tuple(out, arg_out);
}
