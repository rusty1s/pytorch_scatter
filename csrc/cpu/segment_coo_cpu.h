#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
segment_coo_cpu(torch::Tensor src, torch::Tensor index,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size, std::string reduce);

torch::Tensor gather_coo_cpu(torch::Tensor src, torch::Tensor index,
                             std::optional<torch::Tensor> optional_out);
