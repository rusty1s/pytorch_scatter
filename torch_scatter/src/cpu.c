#include <TH/TH.h>

#include "THTensorDimApply.h"

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _, Real)

inline void assertIndexInBoundaries(int idx, int size, long *free) {
  if (idx < 0 || idx >= size) { THFree(free); THError("Invalid index"); }
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
