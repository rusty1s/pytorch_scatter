#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_csr(at::Tensor src, at::Tensor indptr, at::optional<at::Tensor> out_opt,
            std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());
  AT_ASSERTM(false, "Not yet implemented");
  return std::make_tuple(src, at::nullopt);
}

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_coo(at::Tensor src, at::Tensor index, at::Tensor out,
            std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);
  AT_ASSERTM(false, "Not yet implemented");
  return std::make_tuple(src, at::nullopt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_csr", &segment_csr, "Segment CSR (CPU)");
  m.def("segment_coo", &segment_coo, "Segment COO (CPU)");
}
