#include <Python.h>
#include <torch/script.h>

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

int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

static auto registry =
    torch::RegisterOperators().op("torch_scatter::cuda_version", &cuda_version);
