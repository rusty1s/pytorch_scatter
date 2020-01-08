#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"
#include "compat.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff

at::Tensor gather_csr_cuda(at::Tensor src, at::Tensor indptr,
                           at::optional<at::Tensor> out_opt) {

  AT_ASSERTM(src.dim() >= indptr.dim());
  for (int i = 0; i < indptr.dim() - 1; i++)
    AT_ASSERTM(src.size(i) == indptr.size(i));

  src = src.contiguous();
  auto gather_dim = indptr.dim() - 1;
  AT_ASSERTM(src.size(gather_dim) == indptr.size(gather_dim) - 1);

  at::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value().contiguous();
    for (int i = 0; i < out.dim(); i++)
      if (i != gather_dim)
        AT_ASSERTM(src.size(i) == out.size(i));
  } else {
    int64_t *d_gather_size = indptr.flatten()[-1].DATA_PTR<int64_t>();
    int64_t *h_gather_size;
    cudaMemcpy(h_gather_size, d_gather_size, sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    auto sizes = src.sizes().vec();
    sizes[gather_dim] = *h_gather_size;
    out = at::empty(sizes, src.options());
  }

  return out;
}
