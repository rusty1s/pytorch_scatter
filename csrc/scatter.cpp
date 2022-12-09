#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <torch/script.h>

#include "cpu/scatter_cpu.h"
#include "macros.h"
#include "utils.h"

#ifdef WITH_CUDA
#include "cuda/scatter_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__scatter_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__scatter_cpu(void) { return NULL; }
#endif
#endif
#endif

torch::Tensor broadcast(torch::Tensor src, torch::Tensor other, int64_t dim) {
  if (src.dim() == 1)
    for (auto i = 0; i < dim; i++)
      src = src.unsqueeze(0);
  for (auto i = src.dim(); i < other.dim(); i++)
    src = src.unsqueeze(-1);
  src = src.expand(other.sizes().vec());
  return src;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_fw(torch::Tensor src, torch::Tensor index, int64_t dim,
           torch::optional<torch::Tensor> optional_out,
           torch::optional<int64_t> dim_size, std::string reduce) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return scatter_cuda(src, index, dim, optional_out, dim_size, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return scatter_cpu(src, index, dim, optional_out, dim_size, reduce);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ScatterSum : public torch::autograd::Function<ScatterSum> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    dim = dim < 0 ? src.dim() + dim : dim;
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();
    index = broadcast(index, src, dim);
    auto result = scatter_fw(src, index, dim, optional_out, dim_size, "sum");
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
    auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::gather(grad_out, dim, index, false);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

class ScatterMul : public torch::autograd::Function<ScatterMul> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    dim = dim < 0 ? src.dim() + dim : dim;
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();
    index = broadcast(index, src, dim);
    auto result = scatter_fw(src, index, dim, optional_out, dim_size, "mul");
    auto out = std::get<0>(result);
    ctx->save_for_backward({src, index, out});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto src = saved[0];
    auto index = saved[1];
    auto out = saved[2];
    auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::gather(grad_out * out, dim, index, false).div_(src);
    grad_in.masked_fill_(grad_in.isnan(), 0);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

class ScatterMean : public torch::autograd::Function<ScatterMean> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    dim = dim < 0 ? src.dim() + dim : dim;
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    auto old_index = index;

    index = broadcast(index, src, dim);
    auto result = scatter_fw(src, index, dim, optional_out, dim_size, "sum");
    auto out = std::get<0>(result);

    auto ones = torch::ones(old_index.sizes(), src.options());
    result = scatter_fw(ones, old_index,
                        old_index.dim() <= dim ? old_index.dim() - 1 : dim,
                        torch::nullopt, out.size(dim), "sum");
    auto count = std::get<0>(result);
    count.masked_fill_(count < 1, 1);
    count = broadcast(count, out, dim);
    if (out.is_floating_point())
      out.true_divide_(count);
    else
      out.div_(count, "floor");

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
    auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    count = torch::gather(count, dim, index, false);
    auto grad_in = torch::gather(grad_out, dim, index, false);
    grad_in.true_divide_(count);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

class ScatterMin : public torch::autograd::Function<ScatterMin> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    dim = dim < 0 ? src.dim() + dim : dim;
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    index = broadcast(index, src, dim);
    auto result = scatter_fw(src, index, dim, optional_out, dim_size, "min");
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
    auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[dim] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

class ScatterMax : public torch::autograd::Function<ScatterMax> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               torch::optional<Variable> optional_out,
                               torch::optional<int64_t> dim_size) {
    dim = dim < 0 ? src.dim() + dim : dim;
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["src_shape"] = src.sizes();

    index = broadcast(index, src, dim);
    auto result = scatter_fw(src, index, dim, optional_out, dim_size, "max");
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
    auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[dim] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

SCATTER_API torch::Tensor
scatter_sum(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size) {
  return ScatterSum::apply(src, index, dim, optional_out, dim_size)[0];
}

SCATTER_API torch::Tensor
scatter_mul(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size) {
  return ScatterMul::apply(src, index, dim, optional_out, dim_size)[0];
}

SCATTER_API torch::Tensor
scatter_mean(torch::Tensor src, torch::Tensor index, int64_t dim,
             torch::optional<torch::Tensor> optional_out,
             torch::optional<int64_t> dim_size) {
  return ScatterMean::apply(src, index, dim, optional_out, dim_size)[0];
}

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_min(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size) {
  auto result = ScatterMin::apply(src, index, dim, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_max(torch::Tensor src, torch::Tensor index, int64_t dim,
            torch::optional<torch::Tensor> optional_out,
            torch::optional<int64_t> dim_size) {
  auto result = ScatterMax::apply(src, index, dim, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators()
                           .op("torch_scatter::scatter_sum", &scatter_sum)
                           .op("torch_scatter::scatter_mul", &scatter_mul)
                           .op("torch_scatter::scatter_mean", &scatter_mean)
                           .op("torch_scatter::scatter_min", &scatter_min)
                           .op("torch_scatter::scatter_max", &scatter_max);
