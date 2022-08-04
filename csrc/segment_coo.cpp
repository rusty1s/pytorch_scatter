#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <torch/script.h>

#include "cpu/segment_coo_cpu.h"
#include "macros.h"
#include "utils.h"

#ifdef WITH_CUDA
#include "cuda/segment_coo_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__segment_coo_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__segment_coo_cpu(void) { return NULL; }
#endif
#endif
#endif

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_coo_fw(torch::Tensor src, torch::Tensor index,
               torch::optional<torch::Tensor> optional_out,
               torch::optional<int64_t> dim_size, std::string reduce) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return segment_coo_cuda(src, index, optional_out, dim_size, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return segment_coo_cpu(src, index, optional_out, dim_size, reduce);
  }
}

torch::Tensor gather_coo_fw(torch::Tensor src, torch::Tensor index,
                            torch::optional<torch::Tensor> optional_out) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return gather_coo_cuda(src, index, optional_out);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return gather_coo_cpu(src, index, optional_out);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SegmentSumCOO : public torch::autograd::Function<SegmentSumCOO> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_coo_fw(src, index, optional_out, dim_size, "sum");
    auto out = std::get<0>(result);
    ctx->save_for_backward({index});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::empty(src_shape, grad_out.options());
    gather_coo_fw(grad_out, index, grad_in);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class SegmentMeanCOO : public torch::autograd::Function<SegmentMeanCOO> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_coo_fw(src, index, optional_out, dim_size, "mean");
    auto out = std::get<0>(result);
    auto count = std::get<1>(result).value();
    ctx->save_for_backward({index, count});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto count = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::empty(src_shape, grad_out.options());
    gather_coo_fw(grad_out, index, grad_in);
    count = gather_coo_fw(count, index, torch::nullopt);
    for (auto i = 0; i < grad_out.dim() - index.dim(); i++)
      count = count.unsqueeze(-1);
    grad_in.true_divide_(count);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class SegmentMinCOO : public torch::autograd::Function<SegmentMinCOO> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_coo_fw(src, index, optional_out, dim_size, "min");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[index.dim() - 1] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(index.dim() - 1, arg_out, grad_out);
    grad_in =
        grad_in.narrow(index.dim() - 1, 0, src_shape[index.dim() - 1] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class SegmentMaxCOO : public torch::autograd::Function<SegmentMaxCOO> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_coo_fw(src, index, optional_out, dim_size, "max");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[index.dim() - 1] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(index.dim() - 1, arg_out, grad_out);
    grad_in =
        grad_in.narrow(index.dim() - 1, 0, src_shape[index.dim() - 1] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class GatherCOO : public torch::autograd::Function<GatherCOO> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto out = gather_coo_fw(src, index, optional_out);
    ctx->save_for_backward({index});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());

    auto grad_in = torch::zeros(src_shape, grad_out.options());
    segment_coo_fw(grad_out, index, grad_in, torch::nullopt, "sum");
    return {grad_in, Variable(), Variable()};
  }
};

SCATTER_API torch::Tensor
segment_sum_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size) {
  return SegmentSumCOO::apply(src, index, optional_out, dim_size)[0];
}

SCATTER_API torch::Tensor
segment_mean_coo(torch::Tensor src, torch::Tensor index,
                 torch::optional<torch::Tensor> optional_out,
                 torch::optional<int64_t> dim_size) {
  return SegmentMeanCOO::apply(src, index, optional_out, dim_size)[0];
}

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_min_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size) {
  auto result = SegmentMinCOO::apply(src, index, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
segment_max_coo(torch::Tensor src, torch::Tensor index,
                torch::optional<torch::Tensor> optional_out,
                torch::optional<int64_t> dim_size) {
  auto result = SegmentMaxCOO::apply(src, index, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

SCATTER_API torch::Tensor
gather_coo(torch::Tensor src, torch::Tensor index,
           torch::optional<torch::Tensor> optional_out) {
  return GatherCOO::apply(src, index, optional_out)[0];
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_scatter::segment_sum_coo", &segment_sum_coo)
        .op("torch_scatter::segment_mean_coo", &segment_mean_coo)
        .op("torch_scatter::segment_min_coo", &segment_min_coo)
        .op("torch_scatter::segment_max_coo", &segment_max_coo)
        .op("torch_scatter::gather_coo", &gather_coo);
