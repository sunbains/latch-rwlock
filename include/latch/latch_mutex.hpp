// SPDX-License-Identifier: MIT OR Apache-2.0
// NUMA-aware mutex. The top level is a partitioned ticket lock, the node
// level is a classic ticket lock. A holder preferentially passes ownership of the
// top lock to a waiter on its own node, which reduces inter-node lock migration;
// a bound on run length keeps other nodes from starving.
//
// Unlock may be performed by a different thread than lock (reader groups rely on
// it): everything needed to release travels in the mutex, not in the thread.
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>

#include "latch/futex.hpp"
#include "latch/platform.hpp"
#include "latch/runtime.hpp"

namespace latch {

// Top-level waiters sit on a node-level queue head, so waits there span a whole
// run of hand-offs on another node. They get a longer budget before parking.
inline constexpr std::uint32_t top_patience_factor = 32;
// A queued waiter should not give up sooner than the thread ahead of it can wake.
// If it does, it parks too, and every hand-off then needs a wake-up: a self-sustaining
// convoy. Admission control keeps the number of such spinners small.
inline constexpr std::uint32_t queue_patience_factor = 8;
inline constexpr std::uint32_t node_ticket_gap = 8;  // nominal pauses per waiter ahead

// Releases are a plain store with no fence, which is what makes this mutex fast, so a
// release can miss a thread going to sleep at that instant: the thread's check of the
// grant and the release's check of m_parked can pass each other. The window is a few
// nanoseconds, and sleeping is bounded so the worst case is a late wake-up. Exclusion
// comes from the tickets and does not depend on any of this.
inline constexpr long ticket_sleep_limit_ns = 1'000'000;
// Gate sleepers can be numerous, and any later release also wakes them.
inline constexpr long gate_sleep_limit_ns = 100'000'000;

// Node-local ticket lock. Request and grant normally get separate lines, but the
// users of this instance are all on one node, so packing them is cheap.
struct alignas(contended_unit) NodeTicketLock {
  std::atomic<std::uint32_t> m_request{0};
  std::atomic<std::uint32_t> m_grant{0};
  std::atomic<std::uint32_t> m_top_granted{0};  // top lock was passed along
  std::atomic<std::uint32_t> m_parked{0};
  std::int32_t m_local_limit = 0;  // remaining local hand-offs; holder only
  std::uint32_t m_rotation = 0;    // releases since start; holder only
  std::atomic<std::uint32_t> m_gate_sleepers{0};
  std::atomic<std::uint32_t> m_gate_seq{0};
};

// Admission control. A saturated lock can only use a few threads: any more spin
// uselessly, and once threads outnumber CPUs they also preempt the thread whose turn
// it is, which stops a FIFO queue dead. So the queue is kept short and surplus
// threads sleep at a gate, off the CPU, whatever their number. A thread returning
// to a short queue walks straight in, so the running set stays small and cache-warm.
// Every rotation_period releases one sleeper is admitted unconditionally, which
// gives long-term fairness.
inline constexpr std::uint32_t rotation_period = 256;

inline void admit(NodeTicketLock& local) noexcept {
  if constexpr (parking_compiled) {
    const std::uint32_t limit = runtime.m_admission_limit;
    if (limit == 0) return;
    const auto depth = [&local] {
      return local.m_request.load(std::memory_order_seq_cst) - local.m_grant.load(std::memory_order_seq_cst);
    };
    while (depth() >= limit) {
      const std::uint32_t sequence = local.m_gate_seq.load(std::memory_order_acquire);
      local.m_gate_sleepers.fetch_add(1, std::memory_order_seq_cst);
      if (depth() >= limit) {
        futex_wait_bits_for(local.m_gate_seq, sequence, FUTEX_BITSET_MATCH_ANY, gate_sleep_limit_ns);
      }
      local.m_gate_sleepers.fetch_sub(1, std::memory_order_relaxed);
      if (local.m_gate_seq.load(std::memory_order_acquire) != sequence) return;  // admitted
    }
  }
}

// Called after the grant store.
inline void admit_next(NodeTicketLock& local, std::uint32_t next, bool rotate) noexcept {
  if (local.m_gate_sleepers.load(std::memory_order_relaxed) == 0) [[likely]] return;
  const std::uint32_t depth = local.m_request.load(std::memory_order_relaxed) - next;
  if (rotate || depth <= runtime.m_admission_limit / 2) {
    local.m_gate_seq.fetch_add(1, std::memory_order_release);
    futex_wake_one(local.m_gate_seq);
  }
}

// Sleep on a ticket word, waking only for our ticket: the bitset is derived from
// the ticket, so a release wakes its successor rather than the whole queue.
[[nodiscard]] constexpr std::uint32_t ticket_bits(std::uint32_t ticket) noexcept {
  return std::uint32_t{1} << (ticket & 31);
}

inline void park_for_ticket(const std::atomic<std::uint32_t>& grant,
                            std::atomic<std::uint32_t>& parked, std::uint32_t ticket,
                            std::uint32_t bits) noexcept {
  parked.fetch_add(1, std::memory_order_seq_cst);
  for (;;) {
    const std::uint32_t current = grant.load(std::memory_order_seq_cst);
    if (current == ticket) break;
    futex_wait_bits_for(grant, current, bits, ticket_sleep_limit_ns);
  }
  parked.fetch_sub(1, std::memory_order_relaxed);
}

// The grant slots follow the object in memory, sized for this machine's node count:
// construct only where grant_bytes() extra bytes are available (see RwLockCore).
class Mutex {
  // Each grant is the sole occupant of its line: a release wakes only the waiter
  // polling this slot, instead of invalidating every waiter's cache.
  struct alignas(contended_unit) GrantLine {
    std::atomic<std::uint32_t> m_grant{0};
    std::atomic<std::uint32_t> m_parked{0};
  };

 public:
  // Fields a user of the mutex keeps for its holder, so they need no unit of their own.
  struct HolderState {
    std::uint64_t m_sample_began = 0;
    std::uint32_t m_writes = 0;
    std::uint32_t m_sampled_writes = 0;
    std::uint32_t m_sampled_reads = 0;
  };

  Mutex() noexcept {
    for (std::uint32_t slot = 0; slot <= runtime.m_grant_mask; ++slot) new (&grant(slot)) GrantLine{};
  }
  Mutex(const Mutex&) = LATCH_DELETE("a lock has identity");
  Mutex& operator=(const Mutex&) = LATCH_DELETE("a lock has identity");

  [[nodiscard]] static std::size_t grant_bytes() noexcept {
    return sizeof(GrantLine) * (std::size_t{runtime.m_grant_mask} + 1);
  }

  // Returns true if the acquisition had to wait or came via local hand-off.
  bool lock(NodeTicketLock& local) noexcept {
    admit(local);
    const std::uint32_t local_ticket = local.m_request.fetch_add(1, std::memory_order_relaxed);
    await_local(local, local_ticket);

    if (local.m_top_granted.load(std::memory_order_relaxed) != 0) {
      LATCH_CLAIM(m_via == &local);
      local.m_top_granted.store(0, std::memory_order_relaxed);
      return true;
    }

    const std::uint32_t ticket = m_request.fetch_add(1, std::memory_order_relaxed);
    GrantLine& line = grant(ticket);
    if constexpr (prefetchw_compiled) {
      // Our release will store to the next slot: fetch it with intent to write.
      prefetch_for_write(&grant(ticket + 1));
    }
    const bool contended = await_top(line, ticket);
    m_owner_ticket = ticket;
    m_via = &local;
    return contended;
  }

  [[nodiscard]] bool try_lock(NodeTicketLock& local) noexcept {
    std::uint32_t request = local.m_request.load(std::memory_order_acquire);
    if (local.m_grant.load(std::memory_order_acquire) != request) return false;
    if (!local.m_request.compare_exchange_strong(request, request + 1, std::memory_order_acquire,
                                                 std::memory_order_relaxed)) {
      return false;
    }
    // Node lock held, and nothing was queued, so no top-lock pass can be pending.
    std::uint32_t ticket = m_request.load(std::memory_order_acquire);
    if (grant(ticket).m_grant.load(std::memory_order_acquire) == ticket &&
        m_request.compare_exchange_strong(ticket, ticket + 1, std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
      m_owner_ticket = ticket;
      m_via = &local;
      return true;
    }
    publish_local(local, request + 1, false);
    return false;
  }

  void unlock() noexcept {
    NodeTicketLock& local = *m_via;
    const std::uint32_t next = local.m_grant.load(std::memory_order_relaxed) + 1;
    // Someone queued behind us on this node?
    if (local.m_request.load(std::memory_order_relaxed) != next) {
      if (--local.m_local_limit >= 0) {
        local.m_top_granted.store(1, std::memory_order_relaxed);
        const bool rotate = (++local.m_rotation % rotation_period) == 0;
        publish_local(local, next, rotate);
        return;
      }
    }
    // Surrender. A run bound is per run, so it restarts here.
    local.m_local_limit = runtime.m_run_bound;
    const bool rotate = (++local.m_rotation % rotation_period) == 0;
    const std::uint32_t successor = m_owner_ticket + 1;
    GrantLine& line = grant(successor);
    // Node lock first. Once the top lock goes, another node's thread can take it, release
    // it and destroy the lock, and a new lock can be given this NodeState: a late store of
    // a grant would corrupt that lock. What remains late is safe, as the memory is
    // type-stable: reads, and in admit_next a gate wake-up, which at worst admits one
    // extra thread to the new lock's queue.
    local.m_grant.store(next, std::memory_order_release);
    line.m_grant.store(successor, std::memory_order_release);
    if constexpr (parking_compiled) {
      // Touching the words after release is safe: instances are type-stable memory.
      std::atomic_signal_fence(std::memory_order_seq_cst);
      if (line.m_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        futex_wake_bits(line.m_grant, top_bits(successor));
      }
      if (local.m_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        futex_wake_bits(local.m_grant, ticket_bits(next));
      }
      admit_next(local, next, rotate);
    }
  }

  // Holder only. Racy by nature; a late arrival just finds the state we leave.
  [[nodiscard]] bool has_waiters() const noexcept {
    const NodeTicketLock& local = *m_via;
    return local.m_request.load(std::memory_order_relaxed) !=
               local.m_grant.load(std::memory_order_relaxed) + 1 ||
           m_request.load(std::memory_order_relaxed) != m_owner_ticket + 1;
  }

  // May report a false "locked" while the state moves, never a false "unlocked".
  [[nodiscard]] bool is_locked() const noexcept {
    const std::uint32_t ticket = m_request.load(std::memory_order_acquire);
    const std::uint32_t granted = grant(ticket).m_grant.load(std::memory_order_acquire);
    const std::uint32_t verify = m_request.load(std::memory_order_acquire);
    return ((ticket ^ verify) | (granted ^ ticket)) != 0;
  }

  void wait_unlocked() const noexcept {
    std::uint32_t ticket = m_request.load(std::memory_order_acquire);
    for (;;) {
      while (static_cast<std::int32_t>(
                 ticket -
                 grant(ticket).m_grant.load(std::memory_order_acquire)) > 0) {
        cpu_pause();
      }
      const std::uint32_t verify = m_request.load(std::memory_order_acquire);
      if (verify == ticket) return;
      ticket = verify;
    }
  }

 private:

  // Low bits of a top ticket select the slot; the bits above select the waiter.
  [[nodiscard]] static std::uint32_t top_bits(std::uint32_t ticket) noexcept {
    return ticket_bits(ticket >> runtime.m_grant_shift);
  }

  static void publish_local(NodeTicketLock& local, std::uint32_t next, bool rotate) noexcept {
    local.m_grant.store(next, std::memory_order_release);
    if constexpr (parking_compiled) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
      if (local.m_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        futex_wake_bits(local.m_grant, ticket_bits(next));
      }
      admit_next(local, next, rotate);
    }
  }

  // Patience is measured against progress, not total waiting. A long queue that
  // keeps moving is worth spinning on; parking there would turn every hand-off
  // into a wake-up. A queue that stops moving (usually a descheduled thread ahead
  // of us, once threads outnumber CPUs) is not.
  static void await_local(NodeTicketLock& local, std::uint32_t ticket) noexcept {
    std::uint32_t last_seen = local.m_grant.load(std::memory_order_acquire);
    std::uint32_t spent = 0;
    while (last_seen != ticket) {
      std::uint32_t pauses = 1;
      if constexpr (proportional_ticket_compiled) {
        pauses = scaled_pauses((ticket - last_seen) * node_ticket_gap);
      }
      n_pause(pauses);
      const std::uint32_t current = local.m_grant.load(std::memory_order_acquire);
      if (current != last_seen) {
        last_seen = current;
        spent = 0;
        continue;
      }
      spent += pauses;
      if constexpr (parking_compiled) {
        if (runtime.m_patience >= 0 &&
            spent >= std::uint64_t{runtime.m_patience_pauses} * queue_patience_factor) {
          park_for_ticket(local.m_grant, local.m_parked, ticket, ticket_bits(ticket));
          return;
        }
      }
    }
  }

  static bool await_top(GrantLine& line, std::uint32_t ticket) noexcept {
    std::uint32_t spent = 0;
    bool contended = false;
    while (line.m_grant.load(std::memory_order_acquire) != ticket) {
      contended = true;
      pause_for(2);
      ++spent;
      if constexpr (parking_compiled) {
        if (runtime.m_patience >= 0 &&
            spent >= static_cast<std::uint64_t>(runtime.m_patience_pauses) * top_patience_factor) {
          park_for_ticket(line.m_grant, line.m_parked, ticket, top_bits(ticket));
          return true;
        }
      }
    }
    return contended;
  }

  [[nodiscard]] GrantLine& grant(std::uint32_t ticket) const noexcept {
    auto* const lines = reinterpret_cast<GrantLine*>(reinterpret_cast<char*>(const_cast<Mutex*>(this)) + sizeof(Mutex));
    return lines[ticket & runtime.m_grant_mask];
  }

  alignas(contended_unit) std::atomic<std::uint32_t> m_request{0};
  alignas(light_unit) std::uint32_t m_owner_ticket = 0;  // holder only
  NodeTicketLock* m_via = nullptr;  // node lock through which we hold the top

 public:
  HolderState m_holder;  // shares the holder-only unit
};

}  // namespace latch
