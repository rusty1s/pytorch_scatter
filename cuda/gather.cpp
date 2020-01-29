#include <torch/script.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")

torch::Tensor gather_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> out_opt);
torch::Tensor gather_coo_cuda(torch::Tensor src, torch::Tensor index,
                              torch::optional<torch::Tensor> out_opt);

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr,
                         torch::optional<torch::Tensor> out_opt) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return gather_csr_cuda(src, indptr, out_opt);
}

torch::Tensor gather_coo(torch::Tensor src, torch::Tensor index,
                         torch::optional<torch::Tensor> out_opt) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return gather_coo_cuda(src, index, out_opt);
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cuda::gather_csr", &gather_csr)
        .op("torch_scatter_cuda::gather_coo", &gather_coo);
