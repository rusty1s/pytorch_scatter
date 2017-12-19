#include <THC/THC.h>

#include "kernel.h"

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _cuda_, Real)
#define scatter_kernel_(NAME) TH_CONCAT_4(scatter_, NAME, _kernel_, Real)
#define index_backward TH_CONCAT_2(index_backward_cuda_, Real)
#define index_backward_kernel TH_CONCAT_2(index_backward_kernel_, Real)

extern THCState *state;

#include "generic/cuda.c"
#include "THCGenerateFloatType.h"
#include "generic/cuda.c"
#include "THCGenerateDoubleType.h"
#include "generic/cuda.c"
#include "THCGenerateByteType.h"
#include "generic/cuda.c"
#include "THCGenerateCharType.h"
#include "generic/cuda.c"
#include "THCGenerateShortType.h"
#include "generic/cuda.c"
#include "THCGenerateIntType.h"
#include "generic/cuda.c"
#include "THCGenerateLongType.h"
