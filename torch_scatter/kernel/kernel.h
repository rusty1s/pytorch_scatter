#ifdef __cplusplus
extern "C" {
#endif

void scatter_mul_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *input);
void scatter_mul_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input);
void scatter_mul_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *input);
void scatter_mul_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *input);
void scatter_mul_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *input);
void scatter_mul_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *input);
void scatter_mul_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *input);

void scatter_div_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *input);
void scatter_div_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input);
void scatter_div_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *input);
void scatter_div_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *input);
void scatter_div_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *input);
void scatter_div_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *input);
void scatter_div_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *input);

void scatter_mean_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *input,       THCudaTensor *count);
void scatter_mean_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaDoubleTensor *count);
void scatter_mean_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *input,   THCudaByteTensor *count);
void scatter_mean_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *input,   THCudaCharTensor *count);
void scatter_mean_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *input,  THCudaShortTensor *count);
void scatter_mean_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *input,    THCudaIntTensor *count);
void scatter_mean_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *input,    THCudaLongTensor *count);

void scatter_max_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *input, THCudaLongTensor *arg);
void scatter_max_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *input, THCudaLongTensor *arg);

void scatter_min_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *input, THCudaLongTensor *arg);
void scatter_min_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *input, THCudaLongTensor *arg);

void index_backward_kernel_Float (THCState *state, int dim,       THCudaTensor *output, THCudaLongTensor *index,       THCudaTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Double(THCState *state, int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Byte  (THCState *state, int dim,   THCudaByteTensor *output, THCudaLongTensor *index,   THCudaByteTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Char  (THCState *state, int dim,   THCudaCharTensor *output, THCudaLongTensor *index,   THCudaCharTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Short (THCState *state, int dim,  THCudaShortTensor *output, THCudaLongTensor *index,  THCudaShortTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Int   (THCState *state, int dim,    THCudaIntTensor *output, THCudaLongTensor *index,    THCudaIntTensor *grad, THCudaLongTensor *arg);
void index_backward_kernel_Long  (THCState *state, int dim,   THCudaLongTensor *output, THCudaLongTensor *index,   THCudaLongTensor *grad, THCudaLongTensor *arg);

#ifdef __cplusplus
}
#endif
