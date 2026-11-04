// SPDX-License-Identifier: MIT OR Apache-2.0
// Per-node reader indicator and contention-managed fetch-and-add.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "latch/config.hpp"
#include "latch/platform.hpp"
#include "latch/runtime.hpp"

namespace latch {

// Atomic add for contended counters. On failure the witnessed value from the CAS
// is fed straight into one immediate retry, which avoids a reload (and an RTS->RTO
// upgrade). After that the distance between the expected and witnessed values
// estimates the concurrency level, and the back-off is proportional to it.
template <FetchIncrementPolicy policy = fetch_policy>
[[gnu::always_inline]] inline std::uint32_t fetch_increment(std::atomic<std::uint32_t>& target,
                                                            std::uint32_t delta = 1) noexcept {
  if constexpr (policy == FetchIncrementPolicy::xadd) {
    return target.fetch_add(delta, std::memory_order_seq_cst);
  } else {
    if constexpr (prefetchw_compiled) prefetch_for_write(&target);
    std::uint32_t expected = target.load(std::memory_order_relaxed);
    if (target.compare_exchange_strong(expected, expected + delta, std::memory_order_seq_cst,
                                       std::memory_order_relaxed)) [[likely]] {
      return expected;
    }
    std::uint32_t steps = 0;
    if constexpr (policy != FetchIncrementPolicy::tdma) {
      const std::uint32_t witnessed = expected;
      if (target.compare_exchange_strong(expected, witnessed + delta, std::memory_order_seq_cst,
                                         std::memory_order_relaxed)) {
        return witnessed;
      }
    }
    for (;;) {
      const std::uint32_t attempted = expected;
      if (target.compare_exchange_strong(expected, attempted + delta, std::memory_order_seq_cst,
                                         std::memory_order_relaxed)) {
        return attempted;
      }
      const auto gap = std::min<std::uint32_t>(
          static_cast<std::uint32_t>(std::abs(static_cast<std::int32_t>(expected - attempted))), 1024);
      if constexpr (policy == FetchIncrementPolicy::backoff) {
        pause_for((gap + 2) * 13);
      } else if constexpr (policy == FetchIncrementPolicy::tdma) {
        // TDMA-like statistical partition: low bits of the value pick a time slot.
        const std::uint32_t time_slot = expected & 0x7;
        const std::uint32_t duration = (((gap > 2 ? gap - 2 : 0) * time_slot) + steps) * 11;
        ++steps;
        if (duration < 10) continue;  // retry at once with the witnessed value
        pause_for(duration);
      } else {
        // NUMA-aware: favor the node whose turn the value suggests.
        const auto turn = ((expected >> 5) - static_cast<std::uint32_t>(current_node())) &
                          static_cast<std::uint32_t>(runtime.m_node_count - 1);
        if (turn == 0) continue;
        pause_for(turn * 500);
      }
      expected = target.load(std::memory_order_relaxed);
    }
  }
}

// Reader count, split into two monotonic counters. Arriving and departing readers
// touch different words, which lowers CAS failures. Sequestered per node, so
// readers on different nodes generate no inter-node coherence traffic.
struct alignas(coherence_unit) ReaderIndicator {
  std::atomic<std::uint32_t> m_ingress{0};
#if LATCH_SPLIT_INDICATOR
  alignas(coherence_unit)
#endif
  std::atomic<std::uint32_t> m_egress{0};

  // Restricted atomic snapshot: ingress is resampled to ratify the pair.
  [[nodiscard]] std::uint32_t effective_count() const noexcept {
    for (;;) {
      const std::uint32_t in = m_ingress.load(std::memory_order_acquire);
      const std::uint32_t out = m_egress.load(std::memory_order_acquire);
      if (in == m_ingress.load(std::memory_order_acquire)) return in - out;
    }
  }
};

}  // namespace latch
