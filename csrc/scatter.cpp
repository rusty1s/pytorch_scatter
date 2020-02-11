#include <Python.h>
#include <torch/script.h>

// #include "cpu/scatter_cpu.h"
// #include "utils.h"

// #ifdef WITH_CUDA
// #include <cuda.h>
// #include "cuda/scatter_cuda.h"
// #endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__scatter(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_fw(torch::Tensor src, torch::Tensor index, int64_t dim,
           torch::optional<torch::Tensor> optional_out,
           torch::optional<int64_t> dim_size, std::string reduce) {
  return std::make_tuple(src, optional_out);
  // if (src.device().is_cuda()) {
  // #ifdef WITH_CUDA
  //   return scatter_cuda(src, index, dim, optional_out, dim_size, reduce);
  // #else
  //   AT_ERROR("Not compiled with CUDA support");
  // #endif
  // } else {
  //   return scatter_cpu(src, index, dim, optional_out, dim_size, reduce);
  // }
}

static auto registry =
    torch::RegisterOperators().op("torch_scatter::scatter_fw", &scatter_fw);
