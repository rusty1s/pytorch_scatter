#include <torch/script.h>

#include "dim_apply.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")

void scatter_mul(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 int64_t dim) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);
  int64_t elems_per_row = index.size(dim), i, idx;
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_mul", [&] {
    DIM_APPLY3(scalar_t, src, int64_t, index, scalar_t, out, dim, {
      for (i = 0; i < elems_per_row; i++) {
        idx = index_data[i * index_stride];
        out_data[idx * out_stride] *= src_data[i * src_stride];
      }
    });
  });
}

void scatter_div(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 int64_t dim) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);
  int64_t elems_per_row = index.size(dim), i, idx;
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_div", [&] {
    DIM_APPLY3(scalar_t, src, int64_t, index, scalar_t, out, dim, {
      for (i = 0; i < elems_per_row; i++) {
        idx = index_data[i * index_stride];
        out_data[idx * out_stride] /= src_data[i * src_stride];
      }
    });
  });
}

void scatter_max(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 torch::Tensor arg, int64_t dim) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);
  int64_t elems_per_row = index.size(dim), i, idx;
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_max", [&] {
    DIM_APPLY4(scalar_t, src, int64_t, index, scalar_t, out, int64_t, arg, dim,
               {
                 for (i = 0; i < elems_per_row; i++) {
                   idx = index_data[i * index_stride];
                   if (src_data[i * src_stride] >= out_data[idx * out_stride]) {
                     out_data[idx * out_stride] = src_data[i * src_stride];
                     arg_data[idx * arg_stride] = i;
                   }
                 }
               });
  });
}

void scatter_min(torch::Tensor src, torch::Tensor index, torch::Tensor out,
                 torch::Tensor arg, int64_t dim) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);
  CHECK_CPU(arg);
  int64_t elems_per_row = index.size(dim), i, idx;
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter_min", [&] {
    DIM_APPLY4(scalar_t, src, int64_t, index, scalar_t, out, int64_t, arg, dim,
               {
                 for (i = 0; i < elems_per_row; i++) {
                   idx = index_data[i * index_stride];
                   if (src_data[i * src_stride] <= out_data[idx * out_stride]) {
                     out_data[idx * out_stride] = src_data[i * src_stride];
                     arg_data[idx * arg_stride] = i;
                   }
                 }
               });
  });
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cpu::scatter_mul", &scatter_mul)
        .op("torch_scatter_cpu::scatter_div", &scatter_div)
        .op("torch_scatter_cpu::scatter_max", &scatter_max)
        .op("torch_scatter_cpu::scatter_min", &scatter_min);
