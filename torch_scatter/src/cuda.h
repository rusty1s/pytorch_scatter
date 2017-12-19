void scatter_mul_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *input);
void scatter_mul_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input);
void scatter_mul_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *input);
void scatter_mul_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *input);
void scatter_mul_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *input);
void scatter_mul_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *input);
void scatter_mul_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *input);

void scatter_div_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *input);
void scatter_div_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input);
void scatter_div_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *input);
void scatter_div_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *input);
void scatter_div_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *input);
void scatter_div_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *input);
void scatter_div_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *input);

void scatter_mean_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *input, THCudaTensor       *num_output);
void scatter_mean_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaDoubleTensor *num_output);
void scatter_mean_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *input, THCudaByteTensor   *num_output);
void scatter_mean_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *input, THCudaCharTensor   *num_output);
void scatter_mean_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *input, THCudaShortTensor  *num_output);
void scatter_mean_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *input, THCudaIntTensor    *num_output);
void scatter_mean_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *input, THCudaLongTensor   *num_output);

void scatter_max_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *input, THCudaLongTensor *arg_output);
void scatter_max_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *input, THCudaLongTensor *arg_output);

void scatter_min_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *input, THCudaLongTensor *arg_output);
void scatter_min_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *input, THCudaLongTensor *arg_output);

void index_backward_cuda_Float (int dim, THCudaTensor       *output, THCudaLongTensor *index, THCudaTensor       *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Double(int dim, THCudaDoubleTensor *output, THCudaLongTensor *index, THCudaDoubleTensor *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Byte  (int dim, THCudaByteTensor   *output, THCudaLongTensor *index, THCudaByteTensor   *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Char  (int dim, THCudaCharTensor   *output, THCudaLongTensor *index, THCudaCharTensor   *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Short (int dim, THCudaShortTensor  *output, THCudaLongTensor *index, THCudaShortTensor  *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Int   (int dim, THCudaIntTensor    *output, THCudaLongTensor *index, THCudaIntTensor    *grad, THCudaLongTensor *arg_grad);
void index_backward_cuda_Long  (int dim, THCudaLongTensor   *output, THCudaLongTensor *index, THCudaLongTensor   *grad, THCudaLongTensor *arg_grad);
