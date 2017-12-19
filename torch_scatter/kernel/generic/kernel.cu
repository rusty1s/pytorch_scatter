#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void scatter_(mul)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  printf("mul");
}

void scatter_(div)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  printf("div");
}

void scatter_(mean)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCTensor *num_output) {
  printf("mean");
}

void scatter_(max)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
  printf("max");
}

void scatter_(min)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
  printf("min");
}

void index_backward(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *grad, THCudaLongTensor *arg_grad) {
  printf("index_backward");
}

#endif
