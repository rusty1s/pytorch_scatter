#include <TH/TH.h>

#define scatter_(NAME) TH_CONCAT_4(scatter_, NAME, _, Real)
#define check_(NAME) TH_CONCAT_4(check_, NAME, _, Real)

inline void check_inBoundaries(int idx, int size, long *free) {
  if (idx < 0 || idx >= size) { THFree(free); THError("Invalid index"); }
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
