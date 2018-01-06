#define ATOMIC_(NAME) \
template <typename T, size_t n> \
struct TH_CONCAT_3(Atomic, NAME, IntegerImpl); \
\
template<typename T> \
struct TH_CONCAT_3(Atomic, NAME, IntegerImpl)<T, 1> { \
  inline __device__ void operator()(T *address, T val) { \
    uint32_t *address_as_ui = (uint32_t *) (address - ((size_t) address & 3)); \
    uint32_t old = *address_as_ui; \
    uint32_t shift = ((size_t) address & 3) * 8; \
    uint32_t res; \
    uint32_t assumed; \
\
    do { \
      assumed = old; \
      res = OP(val, T((old >> shift) & 0xff)); \
      old = (old & ~(0x000000ff << shift)) | (res << shift); \
      old = atomicCAS(address_as_ui, assumed, old); \
    } while (assumed != old); \
  } \
}; \
\
template<typename T> \
struct TH_CONCAT_3(Atomic, NAME, IntegerImpl)<T, 2> { \
  inline __device__ void operator()(T *address, T val) { \
    uint32_t *address_as_ui = (uint32_t *) ((char *) address - ((size_t) address & 2)); \
    uint32_t old = *address_as_ui; \
    uint32_t res; \
    uint32_t newval; \
    uint32_t assumed; \
\
    do { \
      assumed = old; \
      res = OP(val, (size_t) address & 2 ? T(old >> 16) : T(old & 0xffff)); \
      newval = (size_t) address & 2 ? (old & 0xffff) | (res << 16) : (old & 0xffff0000) | res; \
      old = atomicCAS(address_as_ui, assumed, newval); \
    } while (assumed != old); \
  } \
}; \
\
template<typename T> \
struct TH_CONCAT_3(Atomic, NAME, IntegerImpl)<T, 4> { \
  inline __device__ void operator()(T *address, T val) { \
    uint32_t *address_as_ui = (uint32_t *) address; \
    uint32_t old = *address_as_ui; \
    uint32_t assumed; \
\
    do { \
      assumed = old; \
      old = atomicCAS(address_as_ui, assumed, OP(val, (T) old)); \
    } while (assumed != old); \
  } \
}; \
\
template<typename T> \
struct TH_CONCAT_3(Atomic, NAME, IntegerImpl)<T, 8> { \
  inline __device__ void operator()(T *address, T val) { \
    unsigned long long *address_as_ull = (unsigned long long *) address; \
    unsigned long long old = *address_as_ull; \
    unsigned long long assumed; \
\
    do { \
      assumed = old; \
      old = atomicCAS(address_as_ull, assumed, OP(val, (T) old)); \
    } while (assumed != old); \
  } \
}; \
\
template <typename T, size_t n> \
struct TH_CONCAT_3(Atomic, NAME, DecimalImpl); \
 \
template <typename T> \
struct TH_CONCAT_3(Atomic, NAME, DecimalImpl)<T, 4> { \
  inline __device__ void operator()(T *address, T val) { \
    int *address_as_i = (int *) address; \
    int old = *address_as_i; \
    int assumed; \
\
    do { \
      assumed = old; \
      old = atomicCAS(address_as_i, assumed, __float_as_int(OP(val, __int_as_float(assumed)))); \
    } while (assumed != old); \
  } \
}; \
\
template <typename T> \
struct TH_CONCAT_3(Atomic, NAME, DecimalImpl)<T, 8> { \
  inline __device__ void operator()(T *address, T val) { \
    unsigned long long int *address_as_ull = (unsigned long long int *) address; \
    unsigned long long int old = *address_as_ull; \
    unsigned long long int assumed; \
\
    do { \
      assumed = old; \
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(OP(val, __longlong_as_double(assumed)))); \
    } while (assumed != old); \
  } \
};

#define OP(X, Y) Y + X
ATOMIC_(Add)
#undef OP
static inline __device__ void atomAdd(uint8_t *address, uint8_t val) { AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }
static inline __device__ void atomAdd( int8_t *address,  int8_t val) { AtomicAddIntegerImpl< int8_t, sizeof( int8_t)>()(address, val); }
static inline __device__ void atomAdd(int16_t *address, int16_t val) { AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val); }
static inline __device__ void atomAdd(int32_t *address, int32_t val) { atomicAdd(address, val); }
static inline __device__ void atomAdd(int64_t *address, int64_t val) { AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val); }
static inline __device__ void atomAdd(  float *address,   float val) { atomicAdd(address, val); }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
static inline __device__ void atomAdd( double *address,  double val) { AtomicAddDecimalImpl< double, sizeof( double)>()(address, val); }
#else
static inline __device__ void atomAdd( double *address,  double val) { atomicAdd(address, val); }
#endif

#define OP(X, Y) Y * X
ATOMIC_(Mul)
#undef OP
static inline __device__ void atomMul(uint8_t *address, uint8_t val) { AtomicMulIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }
static inline __device__ void atomMul( int8_t *address,  int8_t val) { AtomicMulIntegerImpl< int8_t, sizeof( int8_t)>()(address, val); }
static inline __device__ void atomMul(int16_t *address, int16_t val) { AtomicMulIntegerImpl<int16_t, sizeof(int16_t)>()(address, val); }
static inline __device__ void atomMul(int32_t *address, int32_t val) { AtomicMulIntegerImpl<int32_t, sizeof(int32_t)>()(address, val); }
static inline __device__ void atomMul(int64_t *address, int64_t val) { AtomicMulIntegerImpl<int64_t, sizeof(int64_t)>()(address, val); }
static inline __device__ void atomMul(  float *address,   float val) { AtomicMulDecimalImpl<  float, sizeof(  float)>()(address, val); }
static inline __device__ void atomMul( double *address,  double val) { AtomicMulDecimalImpl< double, sizeof( double)>()(address, val); }

#define OP(X, Y) Y / X
ATOMIC_(Div)
#undef OP
static inline __device__ void atomDiv(uint8_t *address, uint8_t val) { AtomicDivIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }
static inline __device__ void atomDiv( int8_t *address,  int8_t val) { AtomicDivIntegerImpl< int8_t, sizeof( int8_t)>()(address, val); }
static inline __device__ void atomDiv(int16_t *address, int16_t val) { AtomicDivIntegerImpl<int16_t, sizeof(int16_t)>()(address, val); }
static inline __device__ void atomDiv(int32_t *address, int32_t val) { AtomicDivIntegerImpl<int32_t, sizeof(int32_t)>()(address, val); }
static inline __device__ void atomDiv(int64_t *address, int64_t val) { AtomicDivIntegerImpl<int64_t, sizeof(int64_t)>()(address, val); }
static inline __device__ void atomDiv(  float *address,   float val) { AtomicDivDecimalImpl<  float, sizeof(  float)>()(address, val); }
static inline __device__ void atomDiv( double *address,  double val) { AtomicDivDecimalImpl< double, sizeof( double)>()(address, val); }

#define OP(X, Y) max(Y, X)
ATOMIC_(Max)
#undef OP
static inline __device__ void atomMax(uint8_t *address, uint8_t val) { AtomicMaxIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }
static inline __device__ void atomMax( int8_t *address,  int8_t val) { AtomicMaxIntegerImpl< int8_t, sizeof( int8_t)>()(address, val); }
static inline __device__ void atomMax(int16_t *address, int16_t val) { AtomicMaxIntegerImpl<int16_t, sizeof(int16_t)>()(address, val); }
static inline __device__ void atomMax(int32_t *address, int32_t val) { atomicMax(address, val); }
static inline __device__ void atomMax(int64_t *address, int64_t val) { AtomicMaxIntegerImpl<int64_t, sizeof(int64_t)>()(address, val); }
static inline __device__ void atomMax(  float *address,   float val) { AtomicMaxDecimalImpl<  float, sizeof(  float)>()(address, val); }
static inline __device__ void atomMax( double *address,  double val) { AtomicMaxDecimalImpl< double, sizeof( double)>()(address, val); }

#define OP(X, Y) min(Y, X)
ATOMIC_(Min)
#undef OP
static inline __device__ void atomMin(uint8_t *address, uint8_t val) { AtomicMinIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }
static inline __device__ void atomMin( int8_t *address,  int8_t val) { AtomicMinIntegerImpl< int8_t, sizeof( int8_t)>()(address, val); }
static inline __device__ void atomMin(int16_t *address, int16_t val) { AtomicMinIntegerImpl<int16_t, sizeof(int16_t)>()(address, val); }
static inline __device__ void atomMin(int32_t *address, int32_t val) { atomicMin(address, val); }
static inline __device__ void atomMin(int64_t *address, int64_t val) { AtomicMinIntegerImpl<int64_t, sizeof(int64_t)>()(address, val); }
static inline __device__ void atomMin(  float *address,   float val) { AtomicMinDecimalImpl<  float, sizeof(  float)>()(address, val); }
static inline __device__ void atomMin( double *address,  double val) { AtomicMinDecimalImpl< double, sizeof( double)>()(address, val); }
