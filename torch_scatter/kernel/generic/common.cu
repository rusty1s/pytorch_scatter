#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/common.cu"
#else

void thc_(check)(THCState *state, THCTensor *output, THCudaLongTensor *index, THCTensor *input) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, output, input));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));
  THArgCheck(THCTensor_(nDimension)(state, output) <= MAX_DIMS, 1, "Tensor too large or too many dimensions");
}

TensorInfo<real> thc_(getTensorInfo)(THCState *state, THCTensor *tensor) {
  real *data = THCTensor_(data)(state, tensor);
  int dims = THCTensor_(nDimension)(state, tensor);
  int size[MAX_DIMS]; int stride[MAX_DIMS];
  for (int i = 0; i < dims; i++) {
    size[i] = THCTensor_(size)(state, tensor, i);
    stride[i] = THCTensor_(stride)(state, tensor, i);
  }
  return TensorInfo<real>(data, dims, size, stride);
}

#endif
