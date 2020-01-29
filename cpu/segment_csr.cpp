#include <torch/script.h>

#include "segment_csr_impl.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SegmentSumCSR : public torch::autograd::Function<SegmentSumCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_csr(src, indptr, optional_out, "sum");
    auto out = std::get<0>(result);
    ctx->save_for_backward({indptr});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto src_shape = ctx->saved_data["src_shape"].toIntVector();
    auto grad_in = torch::empty(src_shape, grad_out.options());
    gather_csr(grad_out, indptr, grad_in);

    return {grad_in, Variable(), Variable()};
  }
};

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> optional_out) {
  return SegmentSumCSR::apply(src, indptr, optional_out)[0];
}

static auto registry =
    torch::RegisterOperators("torch_scatter_cpu::segment_csr", &segment_csr)
        .op("torch_scatter_cpu::gather_csr", &gather_csr)
        .op("torch_scatter_cpu::segment_sum_csr", &segment_sum_csr);
