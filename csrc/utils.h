#pragma once

#include <torch/script.h>
#include <vector>

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
  std::vector<int64_t> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); i++)
    result.push_back(list[i]);
  return result;
}
