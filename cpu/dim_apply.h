#pragma once

#include <torch/extension.h>

#define DIM_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, DIM, CODE)  \
  [&] {                                                                        \
    TYPE1 *TENSOR1##_data = TENSOR1.data<TYPE1>();                             \
    auto TENSOR1##_size = TENSOR1.size(DIM);                                   \
    auto TENSOR1##_stride = TENSOR1.stride(DIM);                               \
                                                                               \
    TYPE2 *TENSOR2##_data = TENSOR2.data<TYPE2>();                             \
    auto TENSOR2##_size = TENSOR2.size(DIM);                                   \
    auto TENSOR2##_stride = TENSOR2.stride(DIM);                               \
                                                                               \
    TYPE3 *TENSOR3##_data = TENSOR3.data<TYPE3>();                             \
    auto TENSOR3##_size = TENSOR3.size(DIM);                                   \
    auto TENSOR3##_stride = TENSOR3.stride(DIM);                               \
                                                                               \
    auto dims = TENSOR1.dim();                                                 \
    auto zeros = at::zeros(dims, TENSOR1.options().dtype(at::kLong));          \
    auto counter = zeros.data<int64_t>();                                      \
    bool has_finished = false;                                                 \
                                                                               \
    while (!has_finished) {                                                    \
      CODE;                                                                    \
      if (dims == 1)                                                           \
        break;                                                                 \
                                                                               \
      for (int64_t cur_dim = 0; cur_dim < dims; cur_dim++) {                   \
        if (cur_dim == DIM) {                                                  \
          if (cur_dim == dims - 1) {                                           \
            has_finished = true;                                               \
            break;                                                             \
          }                                                                    \
          continue;                                                            \
        }                                                                      \
                                                                               \
        counter[cur_dim]++;                                                    \
        TENSOR1##_data += TENSOR1.stride(cur_dim);                             \
        TENSOR2##_data += TENSOR2.stride(cur_dim);                             \
        TENSOR3##_data += TENSOR3.stride(cur_dim);                             \
                                                                               \
        if (counter[cur_dim] == TENSOR1.size(cur_dim)) {                       \
          if (cur_dim == dims - 1) {                                           \
            has_finished = true;                                               \
            break;                                                             \
          } else {                                                             \
            TENSOR1##_data -= counter[cur_dim] * TENSOR1.stride(cur_dim);      \
            TENSOR2##_data -= counter[cur_dim] * TENSOR2.stride(cur_dim);      \
            TENSOR3##_data -= counter[cur_dim] * TENSOR3.stride(cur_dim);      \
            counter[cur_dim] = 0;                                              \
          }                                                                    \
        } else                                                                 \
          break;                                                               \
      }                                                                        \
    }                                                                          \
  }()

#define DIM_APPLY4(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, TYPE4,      \
                   TENSOR4, DIM, CODE)                                         \
  [&] {                                                                        \
    TYPE1 *TENSOR1##_data = TENSOR1.data<TYPE1>();                             \
    auto TENSOR1##_size = TENSOR1.size(DIM);                                   \
    auto TENSOR1##_stride = TENSOR1.stride(DIM);                               \
                                                                               \
    TYPE2 *TENSOR2##_data = TENSOR2.data<TYPE2>();                             \
    auto TENSOR2##_size = TENSOR2.size(DIM);                                   \
    auto TENSOR2##_stride = TENSOR2.stride(DIM);                               \
                                                                               \
    TYPE3 *TENSOR3##_data = TENSOR3.data<TYPE3>();                             \
    auto TENSOR3##_size = TENSOR3.size(DIM);                                   \
    auto TENSOR3##_stride = TENSOR3.stride(DIM);                               \
                                                                               \
    TYPE4 *TENSOR4##_data = TENSOR4.data<TYPE4>();                             \
    auto TENSOR4##_size = TENSOR4.size(DIM);                                   \
    auto TENSOR4##_stride = TENSOR4.stride(DIM);                               \
                                                                               \
    auto dims = TENSOR1.dim();                                                 \
    auto zeros = at::zeros(dims, TENSOR1.options().dtype(at::kLong));          \
    auto counter = zeros.data<int64_t>();                                      \
    bool has_finished = false;                                                 \
                                                                               \
    while (!has_finished) {                                                    \
      CODE;                                                                    \
      if (dims == 1)                                                           \
        break;                                                                 \
                                                                               \
      for (int64_t cur_dim = 0; cur_dim < dims; cur_dim++) {                   \
        if (cur_dim == DIM) {                                                  \
          if (cur_dim == dims - 1) {                                           \
            has_finished = true;                                               \
            break;                                                             \
          }                                                                    \
          continue;                                                            \
        }                                                                      \
                                                                               \
        counter[cur_dim]++;                                                    \
        TENSOR1##_data += TENSOR1.stride(cur_dim);                             \
        TENSOR2##_data += TENSOR2.stride(cur_dim);                             \
        TENSOR3##_data += TENSOR3.stride(cur_dim);                             \
        TENSOR4##_data += TENSOR4.stride(cur_dim);                             \
                                                                               \
        if (counter[cur_dim] == TENSOR1.size(cur_dim)) {                       \
          if (cur_dim == dims - 1) {                                           \
            has_finished = true;                                               \
            break;                                                             \
          } else {                                                             \
            TENSOR1##_data -= counter[cur_dim] * TENSOR1.stride(cur_dim);      \
            TENSOR2##_data -= counter[cur_dim] * TENSOR2.stride(cur_dim);      \
            TENSOR3##_data -= counter[cur_dim] * TENSOR3.stride(cur_dim);      \
            TENSOR4##_data -= counter[cur_dim] * TENSOR4.stride(cur_dim);      \
            counter[cur_dim] = 0;                                              \
          }                                                                    \
        } else                                                                 \
          break;                                                               \
      }                                                                        \
    }                                                                          \
  }()
