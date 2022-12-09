#pragma once

#include "extensions.h"

namespace scatter {
SCATTER_API int64_t cuda_version() noexcept;

namespace detail {
SCATTER_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace scatter

SCATTER_API torch::Tensor
scatter_sum(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size);

SCATTER_API torch::Tensor
scatter_mul(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size);

SCATTER_API torch::Tensor
scatter_mean(torch::Tensor src, torch::Tensor index, int64_t dim,
             torch::optional<torch::Tensor> optional_out,
             torch::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_min(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_max(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size);

SCATTER_API torch::Tensor
segment_sum_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size);

SCATTER_API torch::Tensor
segment_mean_coo(torch::Tensor src, torch::Tensor index,
                 torch::optional<torch::Tensor> optional_out,
                 torch::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_min_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_max_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size);

SCATTER_API torch::Tensor
gather_coo(torch::Tensor src, torch::Tensor index,
           torch::optional<torch::Tensor> optional_out);

SCATTER_API torch::Tensor
segment_sum_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out);

SCATTER_API torch::Tensor
segment_mean_csr(torch::Tensor src, torch::Tensor indptr,
                 torch::optional<torch::Tensor> optional_out);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_min_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_max_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out);

SCATTER_API torch::Tensor
gather_csr(torch::Tensor src, torch::Tensor indptr,
           torch::optional<torch::Tensor> optional_out);
