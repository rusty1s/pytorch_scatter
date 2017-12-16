#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void scatter_(add)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] += *(input_data + i * input_stride);
    })
}

void scatter_(sub)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] -= *(input_data + i * input_stride);
    })
}

void scatter_(mul)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] *= *(input_data + i * input_stride);
    })
}

void scatter_(div)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] /= *(input_data + i * input_stride);
    })
}

void scatter_(mean)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THTensor *output_count) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, real, output_count, dim, TH_TENSOR_DIM_APPLY4_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] += *(input_data + i * input_stride);
      output_count_data[idx]++;
    })
}

#endif
