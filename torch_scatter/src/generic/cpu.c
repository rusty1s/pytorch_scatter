#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void scatter_(add)(THTensor *output, THLongTensor *index, THTensor *input, int dim) {
  long idx;
  TH_TENSOR_DIM_APPLY3(real, output, real, input, long, index, dim,
    for (int i = 0; i < THLongTensor_size(index, dim); i++) {
      idx = *(index_data + i * index_stride);
      assertInBoundaries(idx, output_size, TH_TENSOR_DIM_APPLY_counter);
      output_data[idx] += *(input_data + i * input_stride);
    })
}

#endif
