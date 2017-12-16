#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

inline void check_(asserts)(THTensor *output, THLongTensor *index, THTensor *input, int dim) {
  // Assert same dimensionality.
  THArgCheck(dim >= 0 && dim < THTensor_(nDimension)(output), 4, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(input), 2, "Index tensor must have same dimensions as input tensor");
  THArgCheck(THTensor_(nDimension)(input) == THTensor_(nDimension)(output), 3, "Input tensor must have same dimensions as output tensor");

  // Assert same tensor sizes across index and input.
  THLongStorage *indexDims = THLongTensor_newSizeOf(index);
  THArgCheck(THTensor_(isSize)(input, indexDims), 2, "Index tensor must have the same size as input tensor.");
  THLongStorage_free(indexDims);

  // Assert same tensor sizes across input and output apart from specified dimension.
  for (int d = 0; d < THTensor_(nDimension)(output); d++) {
    if (d != dim) THArgCheck(THTensor_(size)(output, d) == THTensor_(size)(input, d), 3, "Input tensor must have same size as output tensor apart from the specified dimension");
  }
}

void scatter_(add)(THTensor *output, THLongTensor *index, THTensor *input, int dim) {
  check_(asserts)(output, index, input, dim); long idx;
  TH_TENSOR_DIM_APPLY3(real, output, real, input, long, index, dim,
    for (int i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      check_inBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] += *(input_data + i * input_stride);
    })
}

#endif
