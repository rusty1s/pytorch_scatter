#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
scatter_cuda(torch::Tensor src, torch::Tensor index, int64_t dim,
             std::optional<torch::Tensor> optional_out,
             std::optional<int64_t> dim_size, std::string reduce);
