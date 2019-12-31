#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor segment_add_csr_cuda(at::Tensor src, at::Tensor indptr,
                                at::optional<at::Tensor> out_opt);
at::Tensor segment_add_coo_cuda(at::Tensor src, at::Tensor index,
                                at::Tensor out);

at::Tensor segment_add_csr(at::Tensor src, at::Tensor indptr,
                           at::optional<at::Tensor> out_opt) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return segment_add_csr_cuda(src, indptr, out_opt);
}

at::Tensor segment_add_coo(at::Tensor src, at::Tensor index, at::Tensor out) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  return segment_add_coo_cuda(src, index, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_add_csr", &segment_add_csr, "Segment Add CSR (CUDA)");
  m.def("segment_add_coo", &segment_add_coo, "Segment Add COO (CUDA)");
}
