#include <torch/script.h>

#include "compat.h"
#include "index_info.h"

#include <vector>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr,
                         torch::optional<torch::Tensor> out_opt) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());

  AT_ASSERTM(src.dim() >= indptr.dim(), "Input mismatch");
  for (int i = 0; i < indptr.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == indptr.size(i), "Input mismatch");

  src = src.contiguous();
  auto gather_dim = indptr.dim() - 1;
  AT_ASSERTM(src.size(gather_dim) == indptr.size(gather_dim) - 1,
             "Input mismatch");

  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != gather_dim)
        AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");
  } else {
    auto sizes = src.sizes().vec();
    sizes[gather_dim] = *indptr.flatten()[-1].DATA_PTR<int64_t>();
    out = torch::empty(sizes, src.options());
  }

  auto N = src.size(gather_dim) * (indptr.numel() / indptr.size(-1));
  auto K = src.numel() / N;
  auto E = out.size(gather_dim);

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "gather_csr", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t row_start, row_end;
    for (int n = 0; n < N; n++) {
      int offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
      row_start = indptr_info.data[offset];
      row_end = indptr_info.data[offset + stride];

      for (int k = 0; k < K; k++) {
        vals[k] = src_data[n * K + k];
      }

      offset = (n / (indptr.size(-1) - 1)) * E * K;
      for (int64_t e = row_start; e < row_end; e++) {
        for (int k = 0; k < K; k++) {
          out_data[offset + e * K + k] = vals[k];
        }
      }
    }
  });

  return out;
}

torch::Tensor gather_coo(torch::Tensor src, torch::Tensor index,
                         torch::optional<torch::Tensor> out_opt) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());

  AT_ASSERTM(src.dim() >= index.dim(), "Input mismatch");
  for (int i = 0; i < index.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == index.size(i), "Input mismatch");

  src = src.contiguous();
  auto gather_dim = index.dim() - 1;

  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < index.dim(); i++)
      AT_ASSERTM(out.size(i) == index.size(i), "Input mismatch");
    for (int i = index.dim() + 1; i < src.dim(); i++)
      AT_ASSERTM(out.size(i) == src.size(i), "Input mismatch");
  } else {
    auto sizes = src.sizes().vec();
    sizes[gather_dim] = index.size(gather_dim);
    out = torch::empty(sizes, src.options());
  }

  auto E_1 = index.numel() / out.size(gather_dim);
  auto E_2 = index.size(gather_dim);
  auto K = out.numel() / index.numel();
  auto N = src.size(gather_dim);

  auto index_info = getTensorInfo<int64_t>(index);
  auto stride = index_info.strides[index_info.dims - 1];
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "gather_coo", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t idx, next_idx;
    for (int e_1 = 0; e_1 < E_1; e_1++) {
      int offset = IndexToOffset<int64_t>::get(e_1 * E_2, index_info);
      idx = index_info.data[offset];

      for (int k = 0; k < K; k++) {
        vals[k] = src_data[e_1 * N * K + idx * K + k];
      }

      for (int e_2 = 0; e_2 < E_2; e_2++) {
        for (int k = 0; k < K; k++) {
          out_data[e_1 * E_2 * K + e_2 * K + k] = vals[k];
        }

        if (e_2 < E_2 - 1) {
          next_idx = index_info.data[offset + (e_2 + 1) * stride];
          assert(idx <= next_idx);

          if (idx != next_idx) {
            idx = next_idx;
            for (int k = 0; k < K; k++) {
              vals[k] = src_data[e_1 * N * K + idx * K + k];
            }
          }
        }
      }
    }
  });

  return out;
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cpu::gather_csr", &gather_csr)
        .op("torch_scatter_cpu::gather_coo", &gather_coo);
