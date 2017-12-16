void scatter_add_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input);
void scatter_add_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input);
void scatter_add_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input);
void scatter_add_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input);
void scatter_add_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input);
void scatter_add_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input);
void scatter_add_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input);

void scatter_sub_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input);
void scatter_sub_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input);
void scatter_sub_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input);
void scatter_sub_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input);
void scatter_sub_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input);
void scatter_sub_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input);
void scatter_sub_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input);

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

void scatter_mean_Float (int dim, THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, THFloatTensor  *output_count);
void scatter_mean_Double(int dim, THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, THDoubleTensor *output_count);
void scatter_mean_Byte  (int dim, THByteTensor   *output, THLongTensor *index, THByteTensor   *input, THByteTensor   *output_count);
void scatter_mean_Char  (int dim, THCharTensor   *output, THLongTensor *index, THCharTensor   *input, THCharTensor   *output_count);
void scatter_mean_Short (int dim, THShortTensor  *output, THLongTensor *index, THShortTensor  *input, THShortTensor  *output_count);
void scatter_mean_Int   (int dim, THIntTensor    *output, THLongTensor *index, THIntTensor    *input, THIntTensor    *output_count);
void scatter_mean_Long  (int dim, THLongTensor   *output, THLongTensor *index, THLongTensor   *input, THLongTensor   *output_count);
