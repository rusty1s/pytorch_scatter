#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

at::Tensor gather_csr(at::Tensor src, at::Tensor indptr,
                      at::optional<at::Tensor> out_opt) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());
  AT_ASSERTM(false, "Not yet implemented");
  return src;
}

at::Tensor gather_coo(at::Tensor src, at::Tensor index,
                      at::optional<at::Tensor> out_opt) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());
  AT_ASSERTM(false, "Not yet implemented");
  return src;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_csr", &gather_csr, "Gather CSR (CPU)");
  m.def("gather_coo", &gather_coo, "Gather COO (CPU)");
}
