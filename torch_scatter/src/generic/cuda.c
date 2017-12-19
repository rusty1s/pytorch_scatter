#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void scatter_(mul)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  scatter_kernel_(mul)(state, dim, output, index, input);
}

void scatter_(div)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  scatter_kernel_(div)(state, dim, output, index, input);
}

void scatter_(mean)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCTensor *num_output) {
  scatter_kernel_(mean)(state, dim, output, index, input, num_output);
}

void scatter_(max)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
  scatter_kernel_(max)(state, dim, output, index, input, arg_output);
}

void scatter_(min)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
  scatter_kernel_(min)(state, dim, output, index, input, arg_output);
}

void index_backward(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *grad, THCudaLongTensor *arg_grad) {
  index_backward_kernel(state, dim, output, index, grad, arg_grad);
}

#endif
