void scatter_add_Float (THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, int dim);
void scatter_add_Double(THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, int dim);
void scatter_add_Byte  (THByteTensor   *output, THLongTensor *index, THByteTensor   *input, int dim);
void scatter_add_Char  (THCharTensor   *output, THLongTensor *index, THCharTensor   *input, int dim);
void scatter_add_Short (THShortTensor  *output, THLongTensor *index, THShortTensor  *input, int dim);
void scatter_add_Int   (THIntTensor    *output, THLongTensor *index, THIntTensor    *input, int dim);
void scatter_add_Long  (THLongTensor   *output, THLongTensor *index, THLongTensor   *input, int dim);
