const int MAX_DIMS = 25;
const int NUM_THREADS = 1024;

inline int GET_BLOCKS(const int n) {
  return (n + NUM_THREADS - 1) / NUM_THREADS;
}

template <typename T>
struct TensorInfo {
  TensorInfo(T *t, int d, int sz[MAX_DIMS], int st[MAX_DIMS]) {
    data = t; dims = d;
    for (int i = 0; i < dims; i++) {
      size[i] = sz[i];
      stride[i] = st[i];
    }
  }

  T *data;
  int dims;
  int size[MAX_DIMS];
  int stride[MAX_DIMS];
};

#define KERNEL_LOOP(I, N) \
  for (int I = blockIdx.x * blockDim.x + threadIdx.x; I < N; i += blockDim.x * gridDim.x)

/* #define KERNEL_RUN(NAME, DIMS, N, PARAMS) \ */
#define KERNEL_RUN(NAME, DIMS, N, ...) \
int grid = GET_BLOCKS(N); \
cudaStream_t stream = THCState_getCurrentStream(state); \
switch (DIMS) { \
  case  1: NAME<real,  1><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
  case  2: NAME<real,  2><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
  case  3: NAME<real,  3><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
  default: NAME<real, -1><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
} \
THCudaCheck(cudaGetLastError());
