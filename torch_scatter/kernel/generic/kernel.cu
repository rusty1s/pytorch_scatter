#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void scatter_(mul)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  thc_(check)(state, output, index, input);
  printf("mul");
}

void scatter_(div)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  thc_(check)(state, output, index, input);
  printf("div");
}

void scatter_(mean)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCTensor *count) {
  thc_(check)(state, output, index, input);
  printf("mean");
}

void scatter_(max)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg) {
  thc_(check)(state, output, index, input);

  const int n = THCudaLongTensor_nElement(state, index);
  TensorInfo<real> outputInfo = thc_(getTensorInfo)(state, output);
  TensorInfo<int64_t> indexInfo = thc_getTensorInfo_Long(state, index);
  TensorInfo<real> inputInfo = thc_(getTensorInfo)(state, input);
  TensorInfo<int64_t> argInfo = thc_getTensorInfo_Long(state, arg);

  KERNEL_RUN(maxKernel, indexInfo.dims, n, outputInfo, indexInfo, inputInfo, argInfo, dim)
  /* KERNEL_RUN(argKernel, indexInfo.dims, n, outputInfo, indexInfo, dim) */
}

void scatter_(min)(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *input, THCudaLongTensor *arg) {
  thc_(check)(state, output, index, input);
  printf("min");
}

void index_backward(THCState *state, int dim, THCTensor *output, THCudaLongTensor *index, THCTensor *grad, THCudaLongTensor *arg) {
  thc_(check)(state, output, index, grad);
  printf("index_backward");
}

#endif
