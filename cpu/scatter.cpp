#include <torch/extension.h>

#include "dim_apply.h"

void scatter_mul(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
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

void scatter_div(at::Tensor src, at::Tensor index, at::Tensor out,
                 int64_t dim) {
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

void scatter_max(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
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

void scatter_min(at::Tensor src, at::Tensor index, at::Tensor out,
                 at::Tensor arg, int64_t dim) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_mul", &scatter_mul, "Scatter Mul (CPU)");
  m.def("scatter_div", &scatter_div, "Scatter Div (CPU)");
  m.def("scatter_max", &scatter_max, "Scatter Max (CPU)");
  m.def("scatter_min", &scatter_min, "Scatter Min (CPU)");
}
