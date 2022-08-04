#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo_cpu(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size, std::string reduce);

torch::Tensor gather_coo_cpu(torch::Tensor src, torch::Tensor index,
                             torch::optional<torch::Tensor> optional_out);
