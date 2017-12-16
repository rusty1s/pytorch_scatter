void scatter_add_Float (THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, int dim);
void scatter_add_Double(THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, int dim);
/* void scatter_add_Half  (THHalfTensor   *output, THLongTensor *index, THHalfTensor   *input, int dim); */
void scatter_add_Byte  (THByteTensor   *output, THLongTensor *index, THByteTensor   *input, int dim);
void scatter_add_Char  (THCharTensor   *output, THLongTensor *index, THCharTensor   *input, int dim);
void scatter_add_Short (THShortTensor  *output, THLongTensor *index, THShortTensor  *input, int dim);
void scatter_add_Int   (THIntTensor    *output, THLongTensor *index, THIntTensor    *input, int dim);
void scatter_add_Long  (THLongTensor   *output, THLongTensor *index, THLongTensor   *input, int dim);

void scatter_sub_Float (THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, int dim);
void scatter_sub_Double(THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, int dim);
/* void scatter_sub_Half  (THHalfTensor   *output, THLongTensor *index, THHalfTensor   *input, int dim); */
void scatter_sub_Byte  (THByteTensor   *output, THLongTensor *index, THByteTensor   *input, int dim);
void scatter_sub_Char  (THCharTensor   *output, THLongTensor *index, THCharTensor   *input, int dim);
void scatter_sub_Short (THShortTensor  *output, THLongTensor *index, THShortTensor  *input, int dim);
void scatter_sub_Int   (THIntTensor    *output, THLongTensor *index, THIntTensor    *input, int dim);
void scatter_sub_Long  (THLongTensor   *output, THLongTensor *index, THLongTensor   *input, int dim);

void scatter_mul_Float (THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, int dim);
void scatter_mul_Double(THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, int dim);
/* void scatter_mul_Half  (THHalfTensor   *output, THLongTensor *index, THHalfTensor   *input, int dim); */
void scatter_mul_Byte  (THByteTensor   *output, THLongTensor *index, THByteTensor   *input, int dim);
void scatter_mul_Char  (THCharTensor   *output, THLongTensor *index, THCharTensor   *input, int dim);
void scatter_mul_Short (THShortTensor  *output, THLongTensor *index, THShortTensor  *input, int dim);
void scatter_mul_Int   (THIntTensor    *output, THLongTensor *index, THIntTensor    *input, int dim);
void scatter_mul_Long  (THLongTensor   *output, THLongTensor *index, THLongTensor   *input, int dim);

void scatter_div_Float (THFloatTensor  *output, THLongTensor *index, THFloatTensor  *input, int dim);
void scatter_div_Double(THDoubleTensor *output, THLongTensor *index, THDoubleTensor *input, int dim);
/* void scatter_div_Half  (THHalfTensor   *output, THLongTensor *index, THHalfTensor   *input, int dim); */
void scatter_div_Byte  (THByteTensor   *output, THLongTensor *index, THByteTensor   *input, int dim);
void scatter_div_Char  (THCharTensor   *output, THLongTensor *index, THCharTensor   *input, int dim);
void scatter_div_Short (THShortTensor  *output, THLongTensor *index, THShortTensor  *input, int dim);
void scatter_div_Int   (THIntTensor    *output, THLongTensor *index, THIntTensor    *input, int dim);
void scatter_div_Long  (THLongTensor   *output, THLongTensor *index, THLongTensor   *input, int dim);
