#pragma once

#ifdef _WIN32
#if defined(torchscatter_EXPORTS)
#define SCATTER_API __declspec(dllexport)
#else
#define SCATTER_API __declspec(dllimport)
#endif
#else
#define SCATTER_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define SCATTER_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define SCATTER_INLINE_VARIABLE __declspec(selectany)
#else
#define SCATTER_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
