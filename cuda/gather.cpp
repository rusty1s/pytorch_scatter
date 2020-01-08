#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor gather_csr_cuda(at::Tensor src, at::Tensor indptr,
                           at::optional<at::Tensor> out_opt);

at::Tensor gather_csr(at::Tensor src, at::Tensor indptr,
                      at::optional<at::Tensor> out_opt) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return gather_csr_cuda(src, indptr, out_opt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_csr", &gather_csr, "Gather CSR (CUDA)");
}
