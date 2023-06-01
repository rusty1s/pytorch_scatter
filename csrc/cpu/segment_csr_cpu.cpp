#include "segment_csr_cpu.h"

#include "index_info.h"
#include "reducer.h"
#include "utils.h"
#include <ATen/OpMathType.h>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out,
                std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

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
    for (auto i = 0; i < out.dim(); i++)
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

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  std::vector<int64_t> args(K);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "segment_csr_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    std::vector<opmath_t> vals(K);
    int64_t row_start, row_end;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (auto n = 0; n < N; n++) {
        auto offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
        row_start = indptr_info.data[offset];
        row_end = indptr_info.data[offset + stride];

        offset = (n / (indptr.size(-1) - 1)) * E * K;
        for (auto k = 0; k < K; k++)
          vals[k] = Reducer<opmath_t, REDUCE>::init();

        for (auto e = row_start; e < row_end; e++)
          for (auto k = 0; k < K; k++)
            Reducer<opmath_t, REDUCE>::update(
                &vals[k], static_cast<opmath_t>(src_data[offset + e * K + k]), &args[k], e);

        for (auto k = 0; k < K; k++)
          Reducer<scalar_t, REDUCE>::write(out_data + n * K + k, static_cast<scalar_t>(vals[k]),
                                           arg_out_data + n * K + k, args[k],
                                           row_end - row_start);
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

torch::Tensor gather_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                             torch::optional<torch::Tensor> optional_out) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

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
    if (src.numel() > 0)
      sizes[dim] = *indptr.flatten()[-1].data_ptr<int64_t>();
    else
      sizes[dim] = 0;
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

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "gather_csr_cpu", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t row_start, row_end;
    for (auto n = 0; n < N; n++) {
      auto offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
      row_start = indptr_info.data[offset];
      row_end = indptr_info.data[offset + stride];

      for (auto k = 0; k < K; k++)
        vals[k] = src_data[n * K + k];

      offset = (n / (indptr.size(-1) - 1)) * E * K;
      for (auto e = row_start; e < row_end; e++)
        for (auto k = 0; k < K; k++)
          out_data[offset + e * K + k] = vals[k];
    }
  });

  return out;
}
