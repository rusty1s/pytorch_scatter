#include <Python.h>
#include <torch/script.h>
#include "scatter.h"
#include "macros.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif

namespace scatter {
SCATTER_API int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}
} // namespace scatter

static auto registry = torch::RegisterOperators().op(
    "torch_scatter::cuda_version", &scatter::cuda_version);
