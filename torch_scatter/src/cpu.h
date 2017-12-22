void scatter_mul_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input);
void scatter_mul_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input);
void scatter_mul_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input);
void scatter_mul_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input);
void scatter_mul_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input);
void scatter_mul_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input);
void scatter_mul_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input);

void scatter_div_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input);
void scatter_div_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input);
void scatter_div_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input);
void scatter_div_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input);
void scatter_div_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input);
void scatter_div_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input);
void scatter_div_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input);

void scatter_mean_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, THFloatTensor  *count);
void scatter_mean_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, THDoubleTensor *count);
void scatter_mean_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input, THByteTensor   *count);
void scatter_mean_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input, THCharTensor   *count);
void scatter_mean_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input, THShortTensor  *count);
void scatter_mean_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input, THIntTensor    *count);
void scatter_mean_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input, THLongTensor   *count);

void scatter_max_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, THLongTensor *arg);
void scatter_max_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, THLongTensor *arg);
void scatter_max_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input, THLongTensor *arg);
void scatter_max_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input, THLongTensor *arg);
void scatter_max_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input, THLongTensor *arg);
void scatter_max_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input, THLongTensor *arg);
void scatter_max_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input, THLongTensor *arg);

void scatter_min_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, THLongTensor *arg);
void scatter_min_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, THLongTensor *arg);
void scatter_min_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input, THLongTensor *arg);
void scatter_min_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input, THLongTensor *arg);
void scatter_min_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input, THLongTensor *arg);
void scatter_min_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input, THLongTensor *arg);
void scatter_min_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input, THLongTensor *arg);

void index_backward_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *grad, THLongTensor *arg);
void index_backward_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *grad, THLongTensor *arg);
void index_backward_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *grad, THLongTensor *arg);
void index_backward_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *grad, THLongTensor *arg);
void index_backward_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *grad, THLongTensor *arg);
void index_backward_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *grad, THLongTensor *arg);
void index_backward_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *grad, THLongTensor *arg);
