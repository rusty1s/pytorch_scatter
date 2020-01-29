#include <torch/script.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                 torch::optional<torch::Tensor> out_opt, std::string reduce);
std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr(torch::Tensor src, torch::Tensor indptr,
            torch::optional<torch::Tensor> out_opt, std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return segment_csr_cuda(src, indptr, out_opt, reduce);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo(torch::Tensor src, torch::Tensor index, torch::Tensor out,
            std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  return segment_coo_cuda(src, index, out, reduce);
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cuda::segment_csr", &segment_csr)
        .op("torch_scatter_cuda::segment_coo", &segment_coo);
