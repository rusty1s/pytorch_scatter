#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void scatter_(add)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] += input_data[i];
    })
}

void scatter_(sub)(int dim, THTensor *output, THLongTensor *index, THTensor *input) {
  TH_TENSOR_DIM_APPLY3(real, output, int64_t, index, real, input, dim, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[index_data[i]] -= input_data[i];
    })
}

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

void scatter_(max)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *output_index) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, output_index, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] >= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        output_index_data[index_data[i]] = i;
      }
    })
}

void scatter_(min)(int dim, THTensor *output, THLongTensor *index, THTensor *input, THLongTensor *output_index) {
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, input, int64_t, output_index, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      assertIndexInBoundaries(index_data[i], output_size, TH_TENSOR_DIM_APPLY_counter);
      if (input_data[i] <= output_data[index_data[i]]) {
        output_data[index_data[i]] = input_data[i];
        output_index_data[index_data[i]] = i;
      }
    })
}

void index_backward(int dim, THTensor *output, THLongTensor *index, THTensor *grad, THLongTensor *grad_index) {
  int64_t idx;
  TH_TENSOR_DIM_APPLY4(real, output, int64_t, index, real, grad, int64_t, grad_index, dim,
    for (int64_t i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      /* if (grad_index_data[index_data[i]] == i) { */
      printf("i: %lli, idx: %lli grad_index: %i grad: %i \n", i, idx, *(grad_index_data + idx * grad_index_stride), *(grad_data + idx * grad_stride));
      /* output_data[i] = grad_data[idx]; */
      /* } */
    })
}

#endif
