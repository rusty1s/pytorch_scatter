#include <TH/TH.h>

#include "THTensorDimApply4.h"

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _, Real)
#define index_backward TH_CONCAT_2(index_backward_, Real)

inline void assertIndexInBoundaries(int idx, int size, int64_t *free) {
  if (idx < 0 || idx >= size) { THFree(free); THError("Invalid index"); }
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
