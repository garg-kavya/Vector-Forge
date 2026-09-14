#pragma once

// Software prefetch hint (no effect on results; a no-op where unsupported).

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace vf::detail {

// Hints that the cache line containing `address` will be read soon.
inline void prefetch_read(const void* address) noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__)
  __builtin_prefetch(address, 0, 3);
#else
  static_cast<void>(address);
#endif
}

}  // namespace vf::detail
