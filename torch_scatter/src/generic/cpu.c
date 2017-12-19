#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void scatter_(mul)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] *= input_data[i];
    })
}

void scatter_(div)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] /= input_data[i];
    })
}

void scatter_(mean)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THTensor *output_count) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, real, output_count, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] += input_data[i];
      output_count_data[index_data[i]]++;
    })
}

void scatter_(max)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *output_arg) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, output_arg, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] >= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        output_arg_data[index_data[i]] = i;
      }
    })
}

void scatter_(min)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *output_arg) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, output_arg, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] <= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        output_arg_data[index_data[i]] = i;
      }
    })
}

void index_backward(int dim, THTensor *output, THLongTensor *index, THTensor *grad, THLongTensor *grad_arg) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, grad, int64_t, grad_arg, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      if (grad_arg_data[index_data[i]] == i) output_data[i] = grad_data[index_data[i]];
    })
}

#endif
