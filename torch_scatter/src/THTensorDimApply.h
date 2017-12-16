#define TH_TENSOR_DIM_APPLY4_SIZE_EQ_EXCEPT_DIM(TENSOR1, TENSOR2, TENSOR3, TENSOR4, DIMENSION) { \
  int shape_check_flag = 0; \
  for(TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) { \
    if (TH_TENSOR_DIM_APPLY_i == DIMENSION) continue; \
    if (TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR2->size[TH_TENSOR_DIM_APPLY_i]) shape_check_flag = 1; \
    if (TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR3->size[TH_TENSOR_DIM_APPLY_i]) shape_check_flag = 1; \
    if (TENSOR1->size[TH_TENSOR_DIM_APPLY_i] != TENSOR4->size[TH_TENSOR_DIM_APPLY_i]) shape_check_flag = 1; \
  } \
  if (shape_check_flag == 1) { \
    THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THDescBuff T4buff = _THSizeDesc(TENSOR4->size, TENSOR3->nDimension); \
    THError("Expected %s %s, %s %s, %s %s and %s %s to have the same size apart from dimension %d", \
            #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str, #TENSOR4, T4buff.str, DIMENSION); \
  } \
}

#define TH_TENSOR_DIM_APPLY4(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, TYPE4, TENSOR4, DIMENSION, SIZE_CHECK, CODE) { \
  TYPE1 *TENSOR1##_data = NULL; \
  int64_t TENSOR1##_stride = 0, TENSOR1##_size = 0; \
  TYPE2 *TENSOR2##_data = NULL; \
  int64_t TENSOR2##_stride = 0, TENSOR2##_size = 0; \
  TYPE3 *TENSOR3##_data = NULL; \
  int64_t TENSOR3##_stride = 0, TENSOR3##_size = 0; \
  TYPE4 *TENSOR4##_data = NULL; \
  int64_t TENSOR4##_stride = 0, TENSOR4##_size = 0; \
\
  int64_t *TH_TENSOR_DIM_APPLY_counter = NULL; \
  int TH_TENSOR_DIM_APPLY_hasFinished = 0; \
  int TH_TENSOR_DIM_APPLY_i; \
\
  if ((DIMENSION < 0) || (DIMENSION >= TENSOR1->nDimension)) \
    THError("invalid dimension %d (expected to be 0 <= dim < %d)", DIMENSION, TENSOR1->nDimension); \
\
  int same_dims = 1; \
  if (TENSOR1->nDimension != TENSOR2->nDimension ) same_dims = 0; \
  if (TENSOR1->nDimension != TENSOR3->nDimension ) same_dims = 0; \
  if (TENSOR1->nDimension != TENSOR4->nDimension ) same_dims = 0; \
\
  if (same_dims == 0) { \
    THDescBuff T1buff = _THSizeDesc(TENSOR1->size, TENSOR1->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THDescBuff T4buff = _THSizeDesc(TENSOR4->size, TENSOR3->nDimension); \
    THError("inconsistent tensor size, expected %s %s, %s %s, %s %s and %s %s to have the same " \
            "number of dimensions", #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str, #TENSOR4, T4.buff.str); \
  } \
\
  SIZE_CHECK(TENSOR1, TENSOR2, TENSOR3, DIMENSION) \
\
  TH_TENSOR_DIM_APPLY_counter = (int64_t*)THAlloc(sizeof(int64_t)*(TENSOR1->nDimension)); \
  for (TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
\
  TENSOR1##_data = (TENSOR1)->storage->data+(TENSOR1)->storageOffset; \
  TENSOR1##_stride = (TENSOR1)->stride[DIMENSION]; \
  TENSOR1##_size = TENSOR1->size[DIMENSION]; \
\
  TENSOR2##_data = (TENSOR2)->storage->data+(TENSOR2)->storageOffset; \
  TENSOR2##_stride = (TENSOR2)->stride[DIMENSION]; \
  TENSOR2##_size = TENSOR2->size[DIMENSION]; \
\
  TENSOR3##_data = (TENSOR3)->storage->data+(TENSOR3)->storageOffset; \
  TENSOR3##_stride = (TENSOR3)->stride[DIMENSION]; \
  TENSOR3##_size = TENSOR3->size[DIMENSION]; \
\
  TENSOR4##_data = (TENSOR4)->storage->data+(TENSOR4)->storageOffset; \
  TENSOR4##_stride = (TENSOR4)->stride[DIMENSION]; \
  TENSOR4##_size = TENSOR4->size[DIMENSION]; \
\
  while (!TH_TENSOR_DIM_APPLY_hasFinished) { \
    CODE \
\
    if (TENSOR1->nDimension == 1) break; \
 \
    for (TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) { \
      if (TH_TENSOR_DIM_APPLY_i == DIMENSION) { \
        if (TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        continue; \
      } \
\
      TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]++; \
      TENSOR1##_data += TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR2##_data += TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR3##_data += TENSOR3->stride[TH_TENSOR_DIM_APPLY_i]; \
      TENSOR4##_data += TENSOR4->stride[TH_TENSOR_DIM_APPLY_i]; \
\
      if (TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] == TENSOR1->size[TH_TENSOR_DIM_APPLY_i]) { \
        if (TH_TENSOR_DIM_APPLY_i == TENSOR1->nDimension-1) { \
          TH_TENSOR_DIM_APPLY_hasFinished = 1; \
          break; \
        } \
        else { \
          TENSOR1##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR1->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR2##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR2->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR3##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR3->stride[TH_TENSOR_DIM_APPLY_i]; \
          TENSOR4##_data -= TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i]*TENSOR4->stride[TH_TENSOR_DIM_APPLY_i]; \
          TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
        } \
      } \
      else break; \
    } \
  } \
  THFree(TH_TENSOR_DIM_APPLY_counter); \
}
