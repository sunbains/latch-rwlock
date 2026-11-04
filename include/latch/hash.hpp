// SPDX-License-Identifier: MIT OR Apache-2.0
// Hashes, branch-free idioms and PRNGs. All arithmetic is unsigned so wrap-around
// is well defined.
#pragma once

#include <atomic>
#include <cstdint>

namespace latch {

// Branch-free !!v: converts a control dependency into a data dependency.
[[nodiscard]] constexpr std::uint32_t normalize(std::uint32_t value) noexcept {
  return (value | (0U - value)) >> 31;
}

// Branch-free value ? ~0 : 0.
[[nodiscard]] constexpr std::uintptr_t normalize_mask(std::uintptr_t value) noexcept {
  constexpr unsigned top_bit = (sizeof(std::uintptr_t) * 8) - 1;
  return std::uintptr_t{0} - ((value | (std::uintptr_t{0} - value)) >> top_bit);
}

// Single-word Wang/Jenkins supplemental hash.
[[nodiscard]] constexpr std::uint32_t spread_hash(std::uint32_t hash) noexcept {
  hash += (hash << 15) ^ 0xFFFFCD7DU;
  hash ^= hash >> 10;
  hash += hash << 3;
  hash ^= hash >> 6;
  hash += (hash << 2) + (hash << 14);
  return hash ^ (hash >> 16);
}

// MurmurHash finalizer stage: every input bit affects every output bit.
[[nodiscard]] constexpr std::uint32_t mix_hash(std::uint32_t hash) noexcept {
  hash ^= hash >> 16;
  hash *= 0x85EBCA6BU;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35U;
  return (hash >> 16) ^ (hash & 0x7FFFFFFFU);
}

[[nodiscard]] constexpr std::uint32_t fnv1a_hash(std::uint32_t value) noexcept {
  return (value ^ 0x811C9DC5U) * 0x01000193U;
}

// Marsaglia shift-xor step. 0 is a fixed point, so it reseeds from a global
// ticket. The ticket is bumped with a relaxed load+store: the racy increment is
// intentional (cheap, benign), but relaxed atomics make the race well defined.
[[nodiscard]] inline std::uint32_t marsaglia_next(std::uint32_t value) noexcept {
  if (value == 0) [[unlikely]] {
    constinit static std::atomic<std::uint32_t> ticket{0x1BADD1CEU};
    const std::uint32_t next = ticket.load(std::memory_order_relaxed) + 1;
    ticket.store(next, std::memory_order_relaxed);
    value = mix_hash(next);  // separate the trajectories of different threads
  }
  value ^= value << 6;
  value ^= value >> 21;
  value ^= value << 7;
  return value;
}

// Thread-local stream. Low-order bits are correlated, so callers discard them.
[[nodiscard]] inline std::uint32_t next_random() noexcept {
  constinit static thread_local std::uint32_t state = 0;
  state = marsaglia_next(state);
  return state & 0x7FFFFFFFU;
}

// Bernoulli trial; probability is scaled by 1024. Repeated trials until the first
// success yield a geometric distribution.
[[nodiscard]] inline bool bernoulli(std::uint32_t probability_1024) noexcept {
  return ((next_random() >> 4) & 0x3FFU) < probability_1024;
}

}  // namespace latch
