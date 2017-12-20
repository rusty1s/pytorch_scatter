template <typename T, size_t n>
struct AtomicMaxIntegerImpl;

template<typename T>
struct AtomicMaxIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui =
        (uint32_t *) (address - ((size_t)address & 3));
    uint32_t old = *address_as_ui;
    uint32_t shift = (((size_t)address & 3) * 8);
    uint32_t sum;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicMaxIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui =
        (uint32_t *) ((char *)address - ((size_t)address & 2));
    uint32_t old = *address_as_ui;
    uint32_t sum;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicMaxIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui = (uint32_t *) (address);
    uint32_t old = *address_as_ui;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicMaxIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    int *address_as_ull = (int*) (address);
    int newval = *address_as_ull;
    atomicMax(address_as_ull, newval);
    /* unsigned long long newval; */
    /* unsigned long long assumed; */

    /* do { */
    /*   assumed = old; */
    /*   newval = val +  (T)old; */
    /*   old = atomicCAS(address_as_ui, assumed, newval); */
    /* } while (assumed != old); */
  }
};

static inline __device__ void atomicMax(uint8_t *address, uint8_t val) {}

static inline __device__ void atomicMax(int8_t *address, int8_t val) {}

static inline __device__ void atomicMax(int16_t *address, int16_t val) {}

static inline __device__ void atomicMax(int64_t *address, int64_t val) {
  AtomicMaxIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}

#ifdef CUDA_HALF_TENSOR
static inline __device__ void atomicMax(half *address, half val) {}
#endif

static inline __device__ void atomicMax(float *address, float val) {
  int *address_as_i = (int *) address;
  int old = *address_as_i;
  int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(max(val, __int_as_float(assumed))));
  } while (assumed != old);
}

static inline __device__  void atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *) address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}
