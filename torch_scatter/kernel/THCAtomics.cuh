#define OP(X, Y) max(X, Y)

template <typename T, size_t n>
struct AtomicIntegerImpl;

template<typename T>
struct AtomicIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui = (uint32_t *) (address - ((size_t) address & 3));
    uint32_t old = *address_as_ui;
    uint32_t shift = (((size_t) address & 3) * 8);
    uint32_t res;
    uint32_t assumed;

    do {
      assumed = old;
      res = OP(val, T((old >> shift) & 0xff));
      old = (old & ~(0x000000ff << shift)) | (res << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui = (uint32_t *) ((char *) address - ((size_t) address & 2));
    uint32_t old = *address_as_ui;
    uint32_t res;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      res = OP(val, (size_t) address & 2 ? T(old >> 16) : T(old & 0xffff));
      newval = (size_t) address & 2 ? (old & 0xffff) | (res << 16) : (old & 0xffff0000) | res;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t *address_as_ull = (uint32_t *) (address);
    uint32_t old = *address_as_ull;
    uint32_t assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, OP(val, (T) old));
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    unsigned long long *address_as_ull = (unsigned long long *) (address);
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, OP(val, (T) old));
    } while (assumed != old);
  }
};

static inline __device__ void atomicMax(uint8_t *address, uint8_t val) {
  AtomicIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}

static inline __device__ void atomicMax(int8_t *address, int8_t val) {
  AtomicIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}

static inline __device__ void atomicMax(int16_t *address, int16_t val) {
  AtomicIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}

static inline __device__ void atomicMax(int64_t *address, int64_t val) {
  AtomicIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
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
    old = atomicCAS(address_as_i, assumed, __float_as_int(OP(val, __int_as_float(assumed))));
  } while (assumed != old);
}

static inline __device__  void atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *) address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(OP(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}
