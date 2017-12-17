#define TH_TENSOR_DIM_APPLY4(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, TYPE4, TENSOR4, DIMENSION, CODE) { \
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
  TH_TENSOR_DIM_APPLY_counter = (int64_t*)THAlloc(sizeof(int64_t)*(TENSOR1->nDimension)); \
\
  for (TH_TENSOR_DIM_APPLY_i = 0; TH_TENSOR_DIM_APPLY_i < TENSOR1->nDimension; TH_TENSOR_DIM_APPLY_i++) { \
    TH_TENSOR_DIM_APPLY_counter[TH_TENSOR_DIM_APPLY_i] = 0; \
  } \
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
