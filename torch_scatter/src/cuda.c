#include <THC/THC.h>

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _cuda_, Real)
#define index_backward TH_CONCAT_2(index_backward_cuda_, Real)

extern THCState *state;

#include "generic/cuda.c"
#include "THCGenerateAllTypes.h"
