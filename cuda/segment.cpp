#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

void segment_add_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);

void segment_add(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  segment_add_cuda(src, index, out, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_add", &segment_add, "Segment Add (CUDA)");
}
