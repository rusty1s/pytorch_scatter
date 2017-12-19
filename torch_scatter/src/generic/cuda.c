#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void scatter_(mul)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
}

void scatter_(div)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
}

void scatter_(mean)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCTensor *num_output) {
}

void scatter_(max)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
}

void scatter_(min)(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg_output) {
}

void index_backward(int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *grad, THCudaLongTensor *arg_grad) {
}

#endif
