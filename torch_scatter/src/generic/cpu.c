#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void scatter_(mul)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx * output_stride] *= *(input_data + i * input_stride);
    })
}

void scatter_(div)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx * output_stride] /= *(input_data + i * input_stride);
    })
}

void scatter_(mean)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THTensor *count) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, real, count, dim,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx * output_stride] += *(input_data + i * input_stride);
      output_data[idx * count_stride]++;
    })
}

void scatter_(max)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *arg) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, arg, dim,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      if (*(input_data + i * input_stride) >= *(output_data + idx * output_stride)) {
        output_data[idx * output_stride] = *(input_data + i * input_stride);
        arg_data[idx * arg_stride] = i;
      }
    })
}

void scatter_(min)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *arg) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, arg, dim,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertIndexInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      if (*(input_data + i * input_stride) <= *(output_data + idx * output_stride)) {
        output_data[idx * output_stride] = *(input_data + i * input_stride);
        arg_data[idx * arg_stride] = i;
      }
    })
}

void index_backward(int dim, THTensor *output, THLongTensor *index, THTensor *grad, THLongTensor *arg) {
  int64_t i, idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, grad, int64_t, arg, dim,
    for (i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      if (*(arg_data + idx * arg_stride) == i) output_data[i * output_stride] = *(grad_data + idx * grad_stride);
    })
}

#endif
