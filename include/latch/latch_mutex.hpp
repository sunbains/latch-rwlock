// SPDX-License-Identifier: MIT OR Apache-2.0
// NUMA-aware writer mutexes, selected by latch::Order. In both, waiters queue in FIFO
// order on a ticket lock per node, and a holder passes the lock directly to a waiter on
// its own node for up to RunBound hand-offs.
//
// FifoMutex (the default): a partitioned ticket lock over the node queues. Nodes take
// strict turns. With no neighbour to hand off to, every release crosses between sockets.
//
// BargingMutex: the lock is one word, so an uncontended acquisition is one atomic rather
// than two, and threads may overtake:
//
//  * An arrival may take a free word if its own node's queue is empty, so waiters are in
//    practice overtaken only from another node. Its check and CAS are not one step, so a
//    neighbour that queues between them can be overtaken once.
//  * During a run of hand-offs the word stays set, so nothing can overtake it.
//  * A queue head polls slowly while another node's owner keeps re-acquiring (up to
//    RemotePoll). Moving the lock between nodes is what costs most, so this lets that node
//    run a batch. A lock that has gone idle is taken at once.
//  * Winners schedule the waiters. Each head registers its node in m_waiting, and whoever
//    holds the lock counts acquisitions: after BargeLimit of them it gives the lock to the
//    next registered node, round-robin, by writing that node's turn in the word. So every
//    overtake brings the waiters' turn closer, and a head waits at most about
//    (nodes - 1) * BargeLimit acquisitions, however long those take.
//
// Unlock may be performed by a different thread than lock (reader groups rely on
// it): everything needed to release travels in the mutex, not in the thread.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "latch/fwd.hpp"
#include "latch/futex.hpp"
#include "latch/platform.hpp"
#include "latch/runtime.hpp"

namespace latch {

// Top-level waiters sit on a node-level queue head, so waits there span a whole
// run of hand-offs on another node. They get a longer budget before parking.
inline constexpr std::uint32_t top_patience_factor = 32;
// BargingMutex: nominal pauses between looks at the word while the owner is on this node.
inline constexpr std::uint32_t local_poll_pauses = 2;
// BargingMutex: waiters expect to sit out whole batches on other nodes, with no progress in sight
// meanwhile. If they sleep through them, each hand-off to a sleeper leaves the lock idle until it
// wakes: 8 threads at 100% writes measured 0.09M to 2.4M ops/s, and steady at 2.7M without that.
inline constexpr std::uint32_t barging_patience_factor = 512;
// BargingMutex: the first poll of a lock owned elsewhere, as a fraction of RemotePoll.
inline constexpr std::uint32_t first_poll_divisor = 8;
// A queued waiter should not give up sooner than the thread ahead of it can wake.
// If it does, it parks too, and every hand-off then needs a wake-up: a self-sustaining
// convoy. Admission control keeps the number of such spinners small.
inline constexpr std::uint32_t queue_patience_factor = 8;
inline constexpr std::uint32_t node_ticket_gap = 8;  // nominal pauses per waiter ahead

// Releases are a plain store with no fence, which is what makes this mutex fast, so a
// release can miss a thread going to sleep at that instant: the thread's check of the
// word and the release's check of m_parked can pass each other. The window is a few
// nanoseconds, and sleeping is bounded so the worst case is a late wake-up. Exclusion
// comes from the word and the tickets, and does not depend on any of this.
inline constexpr long ticket_sleep_limit_ns = 1'000'000;
// Gate sleepers can be numerous, and any later release also wakes them.
inline constexpr long gate_sleep_limit_ns = 100'000'000;

// Node-local ticket lock. Request and grant normally get separate lines, but the
// users of this instance are all on one node, so packing them is cheap.
struct alignas(contended_unit) NodeTicketLock {
  std::atomic<std::uint32_t> m_request{0};
  std::atomic<std::uint32_t> m_grant{0};
  std::atomic<std::uint32_t> m_top_granted{0};  // the holder passed the lock along
  std::atomic<std::uint32_t> m_parked{0};
  std::int32_t m_local_limit = 0;  // remaining local hand-offs; holder only
  std::uint32_t m_rotation = 0;    // releases since start; holder only
  std::uint32_t m_node_index = 0;  // never changes
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

[[nodiscard]] inline std::uint32_t queue_depth(const NodeTicketLock& local) noexcept {
  return local.m_request.load(std::memory_order_seq_cst) - local.m_grant.load(std::memory_order_seq_cst);
}

[[gnu::noinline]] inline void admit_slow(NodeTicketLock& local, std::uint32_t limit) noexcept {
  if constexpr (parking_compiled) {
    const auto depth = [&local] { return queue_depth(local); };
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

inline void admit(NodeTicketLock& local) noexcept {
  if constexpr (parking_compiled) {
    const std::uint32_t limit = runtime.m_admission_limit;
    if (limit != 0 && queue_depth(local) >= limit) [[unlikely]] admit_slow(local, limit);
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

inline void publish_node_grant(NodeTicketLock& local, std::uint32_t next, bool rotate) noexcept {
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
[[gnu::noinline]] inline void await_node_ticket_slow(NodeTicketLock& local, std::uint32_t ticket,
                                                     std::uint32_t patience_factor) noexcept {
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
      if (runtime.m_patience >= 0 && spent >= std::uint64_t{runtime.m_patience_pauses} * patience_factor) {
        park_for_ticket(local.m_grant, local.m_parked, ticket, ticket_bits(ticket));
        return;
      }
    }
  }
}

inline void await_node_ticket(NodeTicketLock& local, std::uint32_t ticket, std::uint32_t patience_factor) noexcept {
  if (local.m_grant.load(std::memory_order_acquire) != ticket) await_node_ticket_slow(local, ticket, patience_factor);
}

// Strict order between nodes, through a partitioned ticket lock.
// The grant slots follow the object in memory, sized for this machine's node count:
// construct only where grant_bytes() extra bytes are available (see RwLockCore).
class FifoMutex {
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

  FifoMutex() noexcept {
    for (std::uint32_t slot = 0; slot <= runtime.m_grant_mask; ++slot) new (&grant(slot)) GrantLine{};
  }
  FifoMutex(const FifoMutex&) = LATCH_DELETE("a lock has identity");
  FifoMutex& operator=(const FifoMutex&) = LATCH_DELETE("a lock has identity");

  [[nodiscard]] static std::size_t grant_bytes() noexcept {
    return sizeof(GrantLine) * (std::size_t{runtime.m_grant_mask} + 1);
  }

  void lock(NodeTicketLock& local) noexcept {
    admit(local);
    const std::uint32_t local_ticket = local.m_request.fetch_add(1, std::memory_order_relaxed);
    await_node_ticket(local, local_ticket, queue_patience_factor);

    if (local.m_top_granted.load(std::memory_order_relaxed) != 0) {
      LATCH_CLAIM(m_via == &local);
      local.m_top_granted.store(0, std::memory_order_relaxed);
      return;
    }

    const std::uint32_t ticket = m_request.fetch_add(1, std::memory_order_relaxed);
    GrantLine& line = grant(ticket);
    if constexpr (prefetchw_compiled) {
      // Our release will store to the next slot: fetch it with intent to write.
      prefetch_for_write(&grant(ticket + 1));
    }
    await_top(line, ticket);
    m_owner_ticket = ticket;
    m_via = &local;
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
    publish_node_grant(local, request + 1, false);
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
        publish_node_grant(local, next, rotate);
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

  // May report a false "locked" while the state moves, never a false "unlocked".
  [[nodiscard]] bool is_locked() const noexcept {
    const std::uint32_t ticket = m_request.load(std::memory_order_acquire);
    const std::uint32_t granted = grant(ticket).m_grant.load(std::memory_order_acquire);
    const std::uint32_t verify = m_request.load(std::memory_order_acquire);
    return ((ticket ^ verify) | (granted ^ ticket)) != 0;
  }

 private:
  // Low bits of a top ticket select the slot; the bits above select the waiter.
  [[nodiscard]] static std::uint32_t top_bits(std::uint32_t ticket) noexcept {
    return ticket_bits(ticket >> runtime.m_grant_shift);
  }

  // Out of line so that an uncontended lock() pays for none of the waiting state.
  static void await_top(GrantLine& line, std::uint32_t ticket) noexcept {
    if (line.m_grant.load(std::memory_order_acquire) != ticket) await_top_slow(line, ticket);
  }

  [[gnu::noinline]] static void await_top_slow(GrantLine& line, std::uint32_t ticket) noexcept {
    std::uint32_t spent = 0;
    while (line.m_grant.load(std::memory_order_acquire) != ticket) {
      pause_for(2);
      ++spent;
      if constexpr (parking_compiled) {
        if (runtime.m_patience >= 0 &&
            spent >= static_cast<std::uint64_t>(runtime.m_patience_pauses) * top_patience_factor) {
          park_for_ticket(line.m_grant, line.m_parked, ticket, top_bits(ticket));
          return;
        }
      }
    }
  }

  [[nodiscard]] GrantLine& grant(std::uint32_t ticket) const noexcept {
    auto* const lines = reinterpret_cast<GrantLine*>(reinterpret_cast<char*>(const_cast<FifoMutex*>(this)) + sizeof(FifoMutex));
    return lines[ticket & runtime.m_grant_mask];
  }

  alignas(contended_unit) std::atomic<std::uint32_t> m_request{0};
  alignas(light_unit) std::uint32_t m_owner_ticket = 0;  // holder only
  NodeTicketLock* m_via = nullptr;  // node lock through which we hold the top

 public:
  HolderState m_holder;  // shares the holder-only unit
};

// Lets threads overtake: see the top of this file.
class BargingMutex {
 public:
  // Fields a user of the mutex keeps for its holder, so they need no unit of their own.
  struct HolderState {
    std::uint64_t m_sample_began = 0;
    std::uint32_t m_writes = 0;
    std::uint32_t m_sampled_writes = 0;
    std::uint32_t m_sampled_reads = 0;
  };

  BargingMutex() noexcept = default;
  BargingMutex(const BargingMutex&) = LATCH_DELETE("a lock has identity");
  BargingMutex& operator=(const BargingMutex&) = LATCH_DELETE("a lock has identity");

  [[nodiscard]] static std::size_t grant_bytes() noexcept { return 0; }

  void lock(NodeTicketLock& local) noexcept {
    if (try_overtake(local)) [[likely]] return;
    admit(local);
    const std::uint32_t ticket = local.m_request.fetch_add(1, std::memory_order_relaxed);
    await_node_ticket(local, ticket, barging_patience_factor);
    if (local.m_top_granted.load(std::memory_order_relaxed) != 0) {
      LATCH_CLAIM(m_holds_ticket && m_via.load(std::memory_order_relaxed) == &local);
      local.m_top_granted.store(0, std::memory_order_relaxed);
      note_acquired();
      return;
    }
    acquire_word(local);
    m_via.store(&local, std::memory_order_relaxed);
    m_holds_ticket = true;
    note_acquired();
  }

  [[nodiscard]] bool try_lock(NodeTicketLock& local) noexcept { return try_overtake(local); }

  void unlock() noexcept {
    NodeTicketLock& local = *m_via.load(std::memory_order_relaxed);
    const std::uint32_t word = word_after_unlock();
    if (!m_holds_ticket) {
      release_word(word);
      return;
    }
    const std::uint32_t next = local.m_grant.load(std::memory_order_relaxed) + 1;
    // Someone queued behind us on this node, and no other node's turn has come?
    if (word == word_free && local.m_request.load(std::memory_order_relaxed) != next) {
      if (--local.m_local_limit >= 0) {
        local.m_top_granted.store(1, std::memory_order_relaxed);
        const bool rotate = (++local.m_rotation % rotation_period) == 0;
        publish_node_grant(local, next, rotate);
        return;
      }
    }
    // Surrender. A run bound is per run, so it restarts here.
    local.m_local_limit = runtime.m_run_bound;
    const bool rotate = (++local.m_rotation % rotation_period) == 0;
    // Node queue first. Once the word goes, another thread can take it, release it and
    // destroy the lock, and a new lock can be given this NodeState: a late store of a
    // grant would corrupt that lock. What remains late is safe, as the memory is
    // type-stable: reads, and in admit_next a gate wake-up, which at worst admits one
    // extra thread to the new lock's queue.
    local.m_grant.store(next, std::memory_order_release);
    release_word(word);
    if constexpr (parking_compiled) {
      if (local.m_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        futex_wake_bits(local.m_grant, ticket_bits(next));
      }
      admit_next(local, next, rotate);
    }
  }

  // Never reports a false "unlocked": a run of hand-offs leaves the word set.
  [[nodiscard]] bool is_locked() const noexcept { return m_locked.load(std::memory_order_acquire) != 0; }

 private:
  [[nodiscard]] bool try_word() noexcept {
    std::uint32_t expected = word_free;
    return m_locked.load(std::memory_order_relaxed) == word_free &&
           m_locked.compare_exchange_strong(expected, word_held, std::memory_order_acquire, std::memory_order_relaxed);
  }

  // Not while this node has a queue: that would overtake a neighbour, and it keeps
  // arrivals off the word's line when the mutex is busy.
  [[nodiscard]] bool try_overtake(NodeTicketLock& local) noexcept {
    if (local.m_request.load(std::memory_order_relaxed) != local.m_grant.load(std::memory_order_relaxed)) return false;
    if constexpr (prefetchw_compiled) {
      // Fetch the line with intent to write, so that the CAS need not upgrade it.
      prefetch_for_write(&m_locked);
    }
    if (!try_word()) return false;
    m_via.store(&local, std::memory_order_relaxed);
    m_holds_ticket = false;
    note_acquired();
    return true;
  }

  // Values of the word. A turn is in the word, not beside it, so that a head going to sleep
  // on the word cannot miss being given the lock: the kernel's check sees the change.
  static constexpr std::uint32_t word_free = 0;
  static constexpr std::uint32_t word_held = 1;
  [[nodiscard]] static constexpr std::uint32_t word_for_turn(std::uint32_t node) noexcept { return node + 2; }

  // The scheduling a winner does: what to leave in the word, which gives the next waiting
  // node its turn if one is due. Holder only. A registered head cannot leave while the lock
  // is held, as a queue cannot be abandoned, so the choice stays valid. A registration this
  // misses is seen by the next holder.
  [[nodiscard]] std::uint32_t word_after_unlock() noexcept {
    const std::uint32_t waiting = m_waiting.load(std::memory_order_relaxed);
    if (waiting == 0) [[likely]] return word_free;
    const std::uint32_t acquired = m_acquired.load(std::memory_order_relaxed);
    if (acquired - m_turn_given_at < runtime.m_barge_limit) return word_free;
    m_turn_given_at = acquired;
    // Round-robin, after the node that last had a turn and not after this holder's: a node
    // that is usually the holder would otherwise keep choosing the same next node.
    static_assert(max_numa_nodes < 32);
    const std::uint32_t after = m_last_turn_node + 1;
    const auto offset = static_cast<std::uint32_t>(std::countr_zero(std::rotr(waiting, static_cast<int>(after))));
    m_last_turn_node = (offset + after) % 32;
    return word_for_turn(m_last_turn_node);
  }

  // The release, whichever value it leaves: no stores after this.
  void release_word(std::uint32_t word) noexcept {
    m_locked.store(word, std::memory_order_release);
    if constexpr (parking_compiled) {
      // Touching the words after release is safe: instances are type-stable memory.
      std::atomic_signal_fence(std::memory_order_seq_cst);
      // All of them: at most one per node, and the turn may be for any.
      if (m_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] futex_wake_all(m_locked);
    }
  }

  // Holder only, and read by heads only to pace themselves.
  void note_acquired() noexcept {
    m_acquired.store(m_acquired.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }

  // The head of a node's queue waits to be given a turn, or for the lock to go idle.
  //
  // Waiting out a busy owner on another node is what lets it run a batch, but a lock that is
  // merely free must not be left idle. So a head first waits one short poll, long enough
  // to see a busy owner come back, and then takes a free word if nobody has acquired it
  // since the last look. While someone has, it leaves the word alone and polls less often:
  // that someone counts towards its turn. So it always gets the lock: either acquisitions
  // continue and bring its turn, or they stop and it sees the lock idle.
  [[gnu::noinline]] void acquire_word(const NodeTicketLock& local) noexcept {
    const std::uint32_t my_turn = word_for_turn(local.m_node_index);
    const std::uint32_t my_bit = std::uint32_t{1} << local.m_node_index;
    m_waiting.fetch_or(my_bit, std::memory_order_relaxed);
    const std::uint32_t local_poll = scaled_pauses(local_poll_pauses);
    const std::uint32_t remote_poll = runtime.m_remote_poll_pauses;
    const std::uint64_t park_budget = std::uint64_t{runtime.m_patience_pauses} * barging_patience_factor;
    std::uint64_t spent = 0;  // in pauses, as are all the budgets here
    std::uint64_t park_at = park_budget;
    std::uint32_t poll = std::max(remote_poll / first_poll_divisor, local_poll);
    std::uint32_t last_acquired = m_acquired.load(std::memory_order_relaxed);
    bool first_look = true;
    for (;;) {
      if (m_locked.load(std::memory_order_acquire) == my_turn) {
        m_locked.store(word_held, std::memory_order_relaxed);
        break;
      }
      const bool owner_here = m_via.load(std::memory_order_relaxed) == &local;
      const std::uint32_t acquired = m_acquired.load(std::memory_order_relaxed);
      const bool reacquired = acquired != last_acquired;
      last_acquired = acquired;
      if ((owner_here || (!first_look && !reacquired)) && try_word()) break;
      first_look = false;
      if (reacquired) poll = std::min(poll * 4, remote_poll);
      const std::uint32_t pauses = owner_here ? local_poll : poll;
      n_pause(pauses);
      spent += pauses;
      if constexpr (parking_compiled) {
        if (runtime.m_patience >= 0 && spent >= park_at) {
          m_parked.fetch_add(1, std::memory_order_seq_cst);
          for (;;) {
            const std::uint32_t word = m_locked.load(std::memory_order_seq_cst);
            if (word == word_free || word == my_turn) break;
            futex_wait_bits_for(m_locked, word, FUTEX_BITSET_MATCH_ANY, ticket_sleep_limit_ns);
          }
          m_parked.fetch_sub(1, std::memory_order_relaxed);
          park_at = spent + park_budget;
        }
      }
    }
    // Under the lock, so it never leaves a stale registration for a holder to act on.
    m_waiting.fetch_and(~my_bit, std::memory_order_relaxed);
  }

  // One line: whoever takes the lock has to own it for the word anyway, so what it writes
  // on acquiring costs no further transfer between nodes.
  alignas(contended_unit) std::atomic<std::uint32_t> m_locked{word_free};
  std::atomic<std::uint32_t> m_waiting{0};   // nodes with a head in acquire_word, as bits
  std::atomic<std::uint32_t> m_parked{0};
  std::atomic<std::uint32_t> m_acquired{0};     // acquisitions; heads read it to pace themselves
  std::atomic<NodeTicketLock*> m_via{nullptr};  // the owner's node, likewise

 public:
  HolderState m_holder;  // holder only

 private:
  std::uint32_t m_turn_given_at = 0;  // holder only: m_acquired at the last turn given
  std::uint32_t m_last_turn_node = 0;  // holder only
  bool m_holds_ticket = false;        // holder only: came through m_via's queue
};

template <Order order>
using MutexFor = std::conditional_t<order == Order::fifo, FifoMutex, BargingMutex>;

}  // namespace latch
