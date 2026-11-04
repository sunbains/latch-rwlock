// SPDX-License-Identifier: MIT OR Apache-2.0
// Low-level pure-spin lock, used only for short allocator critical sections.
#pragma once

#include <atomic>
#include <cstdint>

#include "latch/platform.hpp"

namespace latch {

class SpinLock {
 public:
  constexpr SpinLock() noexcept = default;
  SpinLock(const SpinLock&) = LATCH_DELETE("a lock has identity");
  SpinLock& operator=(const SpinLock&) = LATCH_DELETE("a lock has identity");

  void lock() noexcept {
    // Optimistic: SWAP without a preceding load anticipates success and avoids
    // an RTS->RTO upgrade; also gives the branch predictor a clean fast path.
    if (m_word.exchange(1, std::memory_order_acquire) == 0) [[likely]] return;
    for (;;) {
      if (m_word.exchange(1, std::memory_order_acquire) == 0) return;
      while (m_word.load(std::memory_order_relaxed) != 0) cpu_pause();
    }
  }

  [[nodiscard]] bool try_lock() noexcept {
    return m_word.exchange(1, std::memory_order_acquire) == 0;
  }

  void unlock() noexcept { m_word.store(0, std::memory_order_release); }

  [[nodiscard]] bool is_locked() const noexcept {
    return m_word.load(std::memory_order_relaxed) != 0;
  }

 private:
  std::atomic<std::uint32_t> m_word{0};
};

// An immortal table of locks hashed by address, so short protocols can lock an
// object without storing a lock in it.
class AddressLockTable {
 public:
  [[nodiscard]] SpinLock& lock_for(std::uintptr_t address) noexcept {
    address >>= 6;  // log2 of the nominal cache line
    address ^= address >> 10;
    return m_locks[address & (table_size - 1)].m_lock;
  }

 private:
  static constexpr std::size_t table_size = 1024;
  struct alignas(coherence_unit) Slot {
    SpinLock m_lock;
  };
  Slot m_locks[table_size];
};

}  // namespace latch
