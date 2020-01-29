#include <torch/script.h>

#include "segment_coo_impl.h"

static auto registry =
    torch::RegisterOperators("torch_scatter_cpu::segment_coo", &segment_coo)
        .op("torch_scatter_cpu::gather_coo", &gather_coo);
