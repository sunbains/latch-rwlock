// SPDX-License-Identifier: MIT OR Apache-2.0
// Raw futex and membarrier wrappers. Direct syscalls: the interposition library
// must not depend on libstdc++'s atomic wait machinery.
#pragma once

#include <limits.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>

namespace latch {

inline void futex_wait_bits(const std::atomic<std::uint32_t>& word, std::uint32_t expected,
                            std::uint32_t bits) noexcept {
  static_assert(sizeof(word) == sizeof(std::uint32_t));
  syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(&word),
          FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG, expected, nullptr, nullptr, bits);
}

inline void futex_wake_bits(const std::atomic<std::uint32_t>& word, std::uint32_t bits) noexcept {
  syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(&word),
          FUTEX_WAKE_BITSET | FUTEX_PRIVATE_FLAG, INT_MAX, nullptr, nullptr, bits);
}

inline void futex_wait(const std::atomic<std::uint32_t>& word, std::uint32_t expected) noexcept {
  futex_wait_bits(word, expected, FUTEX_BITSET_MATCH_ANY);
}

// Bounded wait: returns on a wake-up, a changed word, or after the timeout.
inline void futex_wait_bits_for(const std::atomic<std::uint32_t>& word, std::uint32_t expected, std::uint32_t bits,
                                long nanoseconds) noexcept {
  timespec deadline{};
  clock_gettime(CLOCK_MONOTONIC, &deadline);
  deadline.tv_nsec += nanoseconds;
  deadline.tv_sec += deadline.tv_nsec / 1'000'000'000;
  deadline.tv_nsec %= 1'000'000'000;
  syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(&word), FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG, expected,
          &deadline, nullptr, bits);
}

inline void futex_wake_one(const std::atomic<std::uint32_t>& word) noexcept {
  syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(&word), FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1, nullptr,
          nullptr, 0);
}

inline void futex_wake_all(const std::atomic<std::uint32_t>& word) noexcept {
  futex_wake_bits(word, FUTEX_BITSET_MATCH_ANY);
}

// Polite pause for waits that have no wake channel (bias revocation).
inline void short_sleep(long nanoseconds) noexcept {
  timespec duration{0, nanoseconds};
  clock_nanosleep(CLOCK_MONOTONIC, 0, &duration, nullptr);
}

}  // namespace latch
