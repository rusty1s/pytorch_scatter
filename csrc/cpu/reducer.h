#pragma once

#include <limits>
#include <map>

enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mean", MEAN}, {"mul", MUL},
    {"div", DIV}, {"min", MIN},   {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      static constexpr ReductionType REDUCE = SUM;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      static constexpr ReductionType REDUCE = MEAN;                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MUL: {                                                                \
      static constexpr ReductionType REDUCE = MUL;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case DIV: {                                                                \
      static constexpr ReductionType REDUCE = DIV;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      static constexpr ReductionType REDUCE = MIN;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      static constexpr ReductionType REDUCE = MAX;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline void update(scalar_t *val, scalar_t new_val, int64_t *arg,
                            int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline void write(scalar_t *address, scalar_t val,
                           int64_t *arg_address, int64_t arg, int count) {
    if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
      *address = val;
    else if (REDUCE == MEAN)
      *address = val / (scalar_t)(count > 0 ? count : 1);
    else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else
        *address = (scalar_t)0;
    }
  }
};
