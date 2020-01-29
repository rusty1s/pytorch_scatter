#include <torch/script.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")

void scatter_mul_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                      int64_t dim);
void scatter_div_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                      int64_t dim);
void scatter_max_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                      torch::Tensor arg, int64_t dim);
void scatter_min_cuda(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                      torch::Tensor arg, int64_t dim);
void index_backward_cuda(torch::Tensor grad, torch::Tensor index,
                         torch::Tensor arg, torch::Tensor out, int64_t dim);

void scatter_mul(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_mul_cuda(src, index, out, dim);
}

void scatter_div(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  scatter_div_cuda(src, index, out, dim);
}

void scatter_max(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 torch::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_max_cuda(src, index, out, arg, dim);
}

void scatter_min(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 torch::Tensor arg, int64_t dim) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  CHECK_CUDA(arg);
  scatter_min_cuda(src, index, out, arg, dim);
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cuda::scatter_mul", &scatter_mul)
        .op("torch_scatter_cuda::scatter_div", &scatter_div)
        .op("torch_scatter_cuda::scatter_max", &scatter_max)
        .op("torch_scatter_cuda::scatter_min", &scatter_min);
