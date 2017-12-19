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

void scatter_(mean)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THTensor *num_output) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, real, num_output, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] += input_data[i];
      num_output_data[index_data[i]]++;
    })
}

void scatter_(max)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *arg_output) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, arg_output, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] >= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        arg_output_data[index_data[i]] = i;
      }
    })
}

void scatter_(min)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *arg_output) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, arg_output, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] <= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        arg_output_data[index_data[i]] = i;
      }
    })
}

void index_backward(int dim, THTensor *output, THLongTensor *index, THTensor *grad, THLongTensor *arg_grad) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, grad, int64_t, arg_grad, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      if (arg_grad_data[index_data[i]] == i) output_data[i] = grad_data[index_data[i]];
    })
}

#endif
