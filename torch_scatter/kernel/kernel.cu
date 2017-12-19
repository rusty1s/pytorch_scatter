#include <THC/THC.h>

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _kernel_, Real)
#define index_backward TH_CONCAT_2(index_backward_kernel_, Real)

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
