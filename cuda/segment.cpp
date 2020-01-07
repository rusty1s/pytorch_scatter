#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_csr_cuda(at::Tensor src, at::Tensor indptr,
                 at::optional<at::Tensor> out_opt, std::string reduce);
std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_coo_cuda(at::Tensor src, at::Tensor index, at::Tensor out,
                 std::string reduce);

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_csr(at::Tensor src, at::Tensor indptr, at::optional<at::Tensor> out_opt,
            std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (out_opt.has_value())
    CHECK_CUDA(out_opt.value());
  return segment_csr_cuda(src, indptr, out_opt, reduce);
}

std::tuple<at::Tensor, at::optional<at::Tensor>>
segment_coo(at::Tensor src, at::Tensor index, at::Tensor out,
            std::string reduce) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_CUDA(out);
  return segment_coo_cuda(src, index, out, reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_csr", &segment_csr, "Segment CSR (CUDA)");
  m.def("segment_coo", &segment_coo, "Segment COO (CUDA)");
}
