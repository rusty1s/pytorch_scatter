#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor>
segment_add_cuda(at::Tensor src, at::Tensor index, at::Tensor out);

std::tuple<at::Tensor, at::Tensor> segment_add(at::Tensor src, at::Tensor index,
                                               at::Tensor out) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  return segment_add_cuda(src, index, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_add", &segment_add, "Segment Add (CUDA)");
}
