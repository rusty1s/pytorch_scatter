#include "segment_coo_cuda.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo_cuda(torch::Tensor src, torch::Tensor index,
                 torch::optional<torch::Tensor> optional_out,
                 torch::optional<int64_t> dim_size, std::string reduce) {
  return std::make_tuple(src, optional_out);
}

torch::Tensor gather_coo_cuda(torch::Tensor src, torch::Tensor index,
                              torch::optional<torch::Tensor> optional_out) {
  return src;
}
