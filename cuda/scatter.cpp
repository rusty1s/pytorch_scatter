#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

void scatter_mul_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);
void scatter_div_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      int64_t dim);
void scatter_max_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim);
void scatter_min_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                      at::Tensor arg, int64_t dim);
void index_backward_cuda(at::Tensor grad, at::Tensor index, at::Tensor arg,
                         at::Tensor out, int64_t dim);

void scatter_mul(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_mul_cuda(src, index, out, dim);
}

void scatter_div(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_div_cuda(src, index, out, dim);
}

void scatter_max(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_max_cuda(src, index, out, arg, dim);
}

void scatter_min(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_min_cuda(src, index, out, arg, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_mul", &scatter_mul, "Scatter Mul (CUDA)");
  m.def("scatter_div", &scatter_div, "Scatter Div (CUDA)");
  m.def("scatter_max", &scatter_max, "Scatter Max (CUDA)");
  m.def("scatter_min", &scatter_min, "Scatter Min (CUDA)");
}
