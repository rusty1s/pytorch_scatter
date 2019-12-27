#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor segment_add_csr_cuda(at::Tensor src, at::Tensor indptr);
void segment_add_coo_cuda(at::Tensor src, at::Tensor index, at::Tensor out);

void segment_add_thrust_cuda(at::Tensor src, at::Tensor index, at::Tensor out);

at::Tensor segment_add_csr(at::Tensor src, at::Tensor indptr) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  return segment_add_csr_cuda(src, indptr);
}

void segment_add_coo(at::Tensor src, at::Tensor index, at::Tensor out) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  segment_add_coo_cuda(src, index, out);
}

void segment_add_thrust(at::Tensor src, at::Tensor index, at::Tensor out) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  return segment_add_thrust_cuda(src, index, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_add_csr", &segment_add_csr, "Segment Add CSR (CUDA)");
  m.def("segment_add_coo", &segment_add_coo, "Segment Add COO (CUDA)");
  m.def("segment_add_thrust", &segment_add_thrust, "Segment Add Thrust (CUDA)");
}
