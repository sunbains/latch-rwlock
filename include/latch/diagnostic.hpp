// SPDX-License-Identifier: MIT OR Apache-2.0
// Thread-safe, allocation-free diagnostics on stderr, so they cannot interleave
// with or re-enter the application's stdout handling.
#pragma once

#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>

// Mutable process-wide state must exist once per process. Default visibility makes
// the dynamic linker merge the inline variables across shared objects even when
// they are compiled with -fvisibility=hidden. Without this, a writer in one object
// would not see the visible readers registered by another.
#define LATCH_SHARED_STATE [[gnu::visibility("default")]]

namespace latch {

[[gnu::format(printf, 1, 2)]] inline void diagnostic(const char* format, ...) noexcept {
  char buffer[512];
  va_list arguments;
  va_start(arguments, format);
  const int length = std::vsnprintf(buffer, sizeof(buffer), format, arguments);
  va_end(arguments);
  if (length <= 0) return;
  const auto count = std::min(static_cast<std::size_t>(length), sizeof(buffer) - 1);
  [[maybe_unused]] const ssize_t _ = write(STDERR_FILENO, buffer, count);
}

}  // namespace latch
