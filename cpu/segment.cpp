#include <torch/extension.h>

#include "compat.h"
#include "index_info.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

enum ReductionType { ADD, MEAN, MIN, MAX };

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    if (reduce == "add") {                                                     \
      const ReductionType REDUCE = ADD;                                        \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "mean") {                                             \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "min") {                                              \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    } else if (reduce == "max") {                                              \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline scalar_t init() {
    if (REDUCE == MIN) {
      return std::numeric_limits<scalar_t>::max();
    } else if (REDUCE == MAX) {
      return std::numeric_limits<scalar_t>::lowest();
    } else {
      return (scalar_t)0;
    }
  }

  static inline void update(scalar_t *val, scalar_t new_val) {
    if (REDUCE == ADD || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if ((REDUCE == MIN && new_val < *val) ||
               (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
    }
  }

  static inline void update(scalar_t *val, scalar_t new_val, int64_t *arg,
                            int64_t new_arg) {
    if (REDUCE == ADD || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if ((REDUCE == MIN && new_val < *val) ||
               (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline void write(scalar_t *address, scalar_t val,
                           int64_t *arg_address, int64_t arg, int count) {
    if (REDUCE == ADD) {
      *address = val;
    } else if (REDUCE == MEAN) {
      *address = val / (count > 0 ? count : (scalar_t)1);
    } else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else {
        *address = (scalar_t)0;
      }
    }
  }
};

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_csr(at::Tensor src, at::Tensor indptr, at::optional<at::Tensor> out_opt,
            std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (out_opt.has_value())
    CHECK_CPU(out_opt.value());

  AT_ASSERTM(src.dim() >= indptr.dim(), "Input mismatch");

  // Broadcasting `indptr` via `expand`.
  auto sizes = indptr.sizes().vec();
  for (int i = 0; i < indptr.dim() - 1; i++) {
    sizes[i] = src.size(i);
  }
  indptr = indptr.expand(sizes);

  src = src.contiguous();
  auto reduce_dim = indptr.dim() - 1;

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != reduce_dim)
        AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");
    AT_ASSERTM(out.size(reduce_dim) == indptr.size(reduce_dim) - 1,
               "Input mismatch");
  } else {
    sizes = src.sizes().vec();
    sizes[reduce_dim] = indptr.size(reduce_dim) - 1;
    out = at::empty(sizes, src.options());
  }

  at::optional<at::Tensor> arg_out = at::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce == "min" || reduce == "max") {
    arg_out = at::full_like(out, src.size(reduce_dim), indptr.options());
    arg_out_data = arg_out.value().DATA_PTR<int64_t>();
  }

  auto N = out.size(reduce_dim) * (indptr.numel() / indptr.size(-1));
  auto K = out.numel() / N;
  auto E = src.size(reduce_dim);

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_csr", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    scalar_t vals[K];
    int64_t row_start, row_end, args[K];
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (int n = 0; n < N; n++) {
        int offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
        row_start = indptr_info.data[offset];
        row_end = indptr_info.data[offset + stride];

        offset = (n / (indptr.size(-1) - 1)) * E * K;
        for (int k = 0; k < K; k++) {
          vals[k] = Reducer<scalar_t, REDUCE>::init();
        }
        for (int64_t e = row_start; e < row_end; e++) {
          for (int k = 0; k < K; k++) {
            Reducer<scalar_t, REDUCE>::update(
                &vals[k], src_data[offset + e * K + k], &args[k], e);
          }
        }
        for (int k = 0; k < K; k++) {
          Reducer<scalar_t, REDUCE>::write(out_data + n * K + k, vals[k],
                                           arg_out_data + n * K + k, args[k],
                                           row_end - row_start);
        }
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_coo(at::Tensor src, at::Tensor index, at::Tensor out,
            std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_CPU(out);

  AT_ASSERTM(src.dim() >= index.dim(), "Input mismatch");

  // Broadcasting `index` via `expand`.
  auto sizes = index.sizes().vec();
  for (int i = 0; i < index.dim(); i++) {
    sizes[i] = src.size(i);
  }
  index = index.expand(sizes);

  src = src.contiguous();
  out = out.contiguous();
  auto reduce_dim = index.dim() - 1;

  for (int i = 0; i < out.dim(); i++)
    if (i != reduce_dim)
      AT_ASSERTM(src.size(i) == out.size(i), "Input mismatch");

  at::optional<at::Tensor> arg_out = at::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce == "min" || reduce == "max") {
    arg_out = at::full_like(out, src.size(reduce_dim), index.options());
    arg_out_data = arg_out.value().DATA_PTR<int64_t>();
  }

  auto E = index.numel();
  auto E_1 = index.numel() / src.size(reduce_dim);
  auto E_2 = src.size(reduce_dim);
  auto K = src.numel() / index.numel();
  auto N = out.size(reduce_dim);

  auto index_info = getTensorInfo<int64_t>(index);
  auto stride = index_info.strides[index_info.dims - 1];
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "segment_coo", [&] {
    auto src_data = src.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    scalar_t vals[K];
    int64_t idx, next_idx, row_start, args[K];
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (int e_1 = 0; e_1 < E_1; e_1++) {
        int offset = IndexToOffset<int64_t>::get(e_1 * E_2, index_info);
        idx = index_info.data[offset];
        row_start = 0;

        for (int k = 0; k < K; k++) {
          vals[k] = out_data[e_1 * N * K + k];
        }

        for (int e_2 = 0; e_2 < E_2; e_2++) {

          for (int k = 0; k < K; k++) {
            Reducer<scalar_t, REDUCE>::update(
                &vals[k], src_data[e_1 * E_2 * K + e_2 * K + k], &args[k], e_2);
          }

          if (e_2 == E_2 - 1) {
            for (int k = 0; k < K; k++) {
              Reducer<scalar_t, REDUCE>::write(
                  out_data + e_1 * N * K + idx * K + k, vals[k],
                  arg_out_data + e_1 * N * K + idx * K + k, args[k],
                  e_2 + 1 - row_start);
            }
          } else {
            next_idx = index_info.data[offset + (e_2 + 1) * stride];

            if (idx != next_idx) {
              for (int k = 0; k < K; k++) {
                Reducer<scalar_t, REDUCE>::write(
                    out_data + e_1 * N * K + idx * K + k, vals[k],
                    arg_out_data + e_1 * N * K + idx * K + k, args[k],
                    e_2 + 1 - row_start);

                vals[k] = out_data[e_1 * N * K + next_idx * K + k];
              }
              row_start = e_2 + 1;
            }

            idx = next_idx;
          }
        }
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_csr", &segment_csr, "Segment CSR (CPU)");
  m.def("segment_coo", &segment_coo, "Segment COO (CPU)");
}
