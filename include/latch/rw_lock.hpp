// SPDX-License-Identifier: MIT OR Apache-2.0
// NUMA-aware reader-writer lock with writer preference.
//
//   Fast read path   Publish the lock in this thread's row of the visible-reader
//                    table while the lock is biased toward readers. No shared
//                    cache line is written.
//   Slow read path   Increment this node's reader indicator, then check the mode.
//   Write path       NUMA-aware mutex for writer-writer exclusion, then drain the
//                    indicators, then revoke the bias if it is set.
//
// Readers cannot starve: a reader that finds a writer in the way retracts, and the
// first of them becomes a leader that queues on the write mutex like a writer. On
// getting it, the leader opens the lock for the group that gathered meanwhile. The
// last of the group to pass keeps the write mutex until its own unlock.
//
// Thread count may exceed CPU count. Waiting spins only while the lock is making
// progress and then parks, surplus writers sleep at an admission gate instead of
// joining a queue, and the read fast path has no shared state to contend on.
#pragma once

#include <errno.h>
#include <time.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <system_error>
#include <type_traits>

#include "latch/fwd.hpp"
#include "latch/latch_mutex.hpp"
#include "latch/futex.hpp"
#include "latch/reader_indicator.hpp"
#include "latch/runtime.hpp"

namespace latch {

// Readers and node lock are separate units: reader traffic must not disturb the
// ticket lock. Allocated per node so each can be homed where it is used.
struct NodeState {
  ReaderIndicator m_readers;
  NodeTicketLock m_ticket;
};

// Readers waiting for a group must not spin long: there can be many of them, and
// each spinner denies a CPU to the thread that would make progress.
inline constexpr std::uint32_t wait_patience_factor = 1;

// Readers gathered behind a leader park almost at once. If they all keep spinning,
// each write stops every reader and then releases them together, to miss on the same
// freshly written lines at the same moment. That regime is stable, and measured 5x
// slower with 30x the memory-ordering machine clears on a 2x24 core Cascade Lake.
inline constexpr std::uint32_t group_patience_divisor = 64;


// Hard configuration: the expected share of acquisitions that are writes, in percent.
// Only a performance hint; a wrong value cannot affect correctness.
struct ExpectedWrites {
  std::uint32_t m_percent;
};

enum class LoadProfile : std::uint32_t { read_mostly, mixed, write_heavy };

// Write share, in percent, at which each profile starts, and below which automatic
// adjustment leaves it. The gaps prevent flapping around one threshold. Measured on
// 2x14 core Broadwell: under 3% readers should park at once; by 30% the fast path's
// revocations cost more than it saves.
inline constexpr std::uint32_t mixed_enter_percent = 3;
inline constexpr std::uint32_t mixed_leave_percent = 2;
inline constexpr std::uint32_t heavy_enter_percent = 30;
inline constexpr std::uint32_t heavy_leave_percent = 20;

// Automatic adjustment samples at most this often, and looks at the clock only every
// sample_write_interval writes. A sample reads one word per thread while holding the
// write mutex, so this keeps it far below 1% of write time.
inline constexpr std::uint64_t sample_interval_cycles = 25'000'000;
inline constexpr std::uint32_t sample_write_interval = 256;

// Write levels live in the spare bits of RecursiveRwLock's word. Past this, nesting is
// refused, as more would overwrite the pointer.
inline constexpr std::uint32_t max_write_nesting = 1024;

enum class LockMode : std::uint32_t {
  reading = 0,
  writer_resolving = 2,  // stays set across a stream of writers: they skip the drain
};

class alignas(coherence_unit) RwLockCore {
 public:
  RwLockCore() noexcept {
    if (runtime.m_expected_writes >= 0) {
      fix_profile(static_cast<std::uint32_t>(runtime.m_expected_writes));
    } else {
      m_adaptive = bias_compiled && runtime.m_bias_enabled;
    }
    apply_profile(m_profile.load(std::memory_order_relaxed));
    for (int node = 0; node < runtime.m_node_count; ++node) {
      m_nodes[node] = new (pool_allocate(PoolKind::node_state, sizeof(NodeState), node)) NodeState{};
      m_nodes[node]->m_ticket.m_local_limit = runtime.m_run_bound;
    }
    constinit static std::atomic<std::uint32_t> next_column{0};
    m_bias_column = next_column.fetch_add(1, std::memory_order_relaxed) % bias_columns;
  }

  ~RwLockCore() {
    for (int node = 0; node < runtime.m_node_count; ++node) {
      m_nodes[node]->~NodeState();
      pool_release(PoolKind::node_state, m_nodes[node], node);
    }
  }

  RwLockCore(const RwLockCore&) = LATCH_DELETE("a lock has identity");
  RwLockCore& operator=(const RwLockCore&) = LATCH_DELETE("a lock has identity");

  explicit RwLockCore(ExpectedWrites expected) noexcept : RwLockCore() {
    m_adaptive = false;
    fix_profile(expected.m_percent);
    apply_profile(m_profile.load(std::memory_order_relaxed));
  }

  [[nodiscard]] static RwLockCore* create() noexcept {
    ensure_runtime();
    return new (pool_allocate(PoolKind::lock_core, allocation_size())) RwLockCore{};
  }
  [[nodiscard]] static RwLockCore* create(ExpectedWrites expected) noexcept {
    ensure_runtime();
    return new (pool_allocate(PoolKind::lock_core, allocation_size())) RwLockCore{expected};
  }

  // Total memory of one lock, including its per-node state.
  [[nodiscard]] static std::size_t footprint() noexcept {
    ensure_runtime();
    return allocation_size() + sizeof(NodeState) * static_cast<std::size_t>(runtime.m_node_count);
  }

  [[nodiscard]] LoadProfile load_profile() const noexcept {
    return m_profile.load(std::memory_order_relaxed);
  }
  [[nodiscard]] bool is_adaptive() const noexcept { return m_adaptive; }
  static void dispose(RwLockCore* core) noexcept {
    core->~RwLockCore();
    pool_release(PoolKind::lock_core, core);
  }

  // Returns 0 or EDEADLK.
  [[gnu::always_inline]] int read_acquire(ThreadState& self) noexcept {
    if constexpr (bias_compiled) {
      if (BiasRow* const row = self.m_bias_row; row != nullptr) [[likely]] {
        auto& slot = row->m_slots[m_bias_column];
        const void* const held = slot.load(std::memory_order_relaxed);
        if (held == this) {
          ++self.m_bias_nesting[m_bias_column];
          return 0;
        }
        if (held == nullptr && m_bias.load(std::memory_order_relaxed) != 0) {
          slot.store(this, std::memory_order_relaxed);
          publish_fence();
          if (m_bias.load(std::memory_order_acquire) != 0) [[likely]] {
            count_read(*row);
            return 0;
          }
          slot.store(nullptr, std::memory_order_relaxed);
        }
      }
    }
    return read_acquire_slow(self);
  }

  // Releases read or write permission, whichever this thread holds.
  [[gnu::always_inline]] int release(ThreadState& self) noexcept {
    if constexpr (bias_compiled) {
      if (BiasRow* const row = self.m_bias_row; row != nullptr) [[likely]] {
        auto& slot = row->m_slots[m_bias_column];
        if (slot.load(std::memory_order_relaxed) == this) [[likely]] {
          std::uint32_t& nesting = self.m_bias_nesting[m_bias_column];
          if (nesting != 0) {
            --nesting;
          } else {
            slot.store(nullptr, std::memory_order_release);
          }
          return 0;
        }
      }
    }
    return release_slow(self);
  }

  // Returns 0 or EDEADLK.
  int write_acquire(ThreadState& self) noexcept {
    if (held_by(self)) {
      if (runtime.m_warn) diagnostic("Warning: wrlock EDEADLK on %p\n", static_cast<void*>(this));
      return EDEADLK;
    }
    m_write_mutex.lock(local_ticket());
    if (m_mode.load(std::memory_order_relaxed) == LockMode::reading) {
      m_mode.store(LockMode::writer_resolving, std::memory_order_relaxed);
      store_load_fence();  // Dekker with the reader's increment-then-check
      drain_readers();
    }
    if (m_bias.load(std::memory_order_relaxed) != 0) revoke_bias();
    if (m_adaptive && (++m_write_mutex.m_holder.m_writes % sample_write_interval) == 0) [[unlikely]] sample_load();
    LATCH_CLAIM(m_writer_active.load(std::memory_order_relaxed) == nullptr);
    m_writer_active.store(&self, std::memory_order_relaxed);
    return 0;
  }

  // Returns 0 or EBUSY.
  int try_read_acquire(ThreadState& self) noexcept {
    if constexpr (bias_compiled) {
      if (BiasRow* const row = self.m_bias_row; row != nullptr) [[likely]] {
        auto& slot = row->m_slots[m_bias_column];
        const void* const held = slot.load(std::memory_order_relaxed);
        if (held == this) {
          ++self.m_bias_nesting[m_bias_column];
          return 0;
        }
        if (held == nullptr && m_bias.load(std::memory_order_relaxed) != 0) {
          slot.store(this, std::memory_order_relaxed);
          publish_fence();
          if (m_bias.load(std::memory_order_acquire) != 0) {
            count_read(*row);
            return 0;
          }
          slot.store(nullptr, std::memory_order_relaxed);
        }
      }
    }
    if (is_write_holder(self)) return EBUSY;
    ReadMemo& memo = self.memo_for(this);
    if (++memo.m_count > 1) return 0;
    ReaderIndicator& readers = m_nodes[memo.m_node]->m_readers;
    fetch_increment(readers.m_ingress);
    if (m_mode.load(std::memory_order_seq_cst) == LockMode::reading) {
      maybe_enable_bias();
      return 0;
    }
    depart(readers);
    // The mode may just be left over from a finished writer. If the write mutex is
    // free nobody needs it set, so open the lock as a group leader would.
    if (m_write_mutex.try_lock(local_ticket())) {
      m_mode.store(LockMode::reading, std::memory_order_release);
      fetch_increment(readers.m_ingress);
      m_write_mutex.unlock();
      return 0;
    }
    memo.m_count = 0;
    memo.m_lock = nullptr;
    return EBUSY;
  }

  // Returns 0 or EBUSY. Never waits for readers.
  int try_write_acquire(ThreadState& self) noexcept {
    if (held_by(self)) return EBUSY;
    if (!m_write_mutex.try_lock(local_ticket())) return EBUSY;
    const bool opened = m_mode.load(std::memory_order_relaxed) == LockMode::reading;
    if (opened) {
      m_mode.store(LockMode::writer_resolving, std::memory_order_relaxed);
      store_load_fence();
      if (any_indicator_busy() || !try_revoke_bias()) {
        m_mode.store(LockMode::reading, std::memory_order_release);
        m_write_mutex.unlock();
        return EBUSY;
      }
    }
    m_writer_active.store(&self, std::memory_order_relaxed);
    return 0;
  }

  [[nodiscard]] bool held_by(const ThreadState& self) const noexcept {
    if (is_write_holder(self)) return true;
    if (self.m_bias_row != nullptr &&
        self.m_bias_row->m_slots[m_bias_column].load(std::memory_order_relaxed) == this) {
      return true;
    }
    return self.find_memo(this) != nullptr;
  }

  [[nodiscard]] bool is_write_holder(const ThreadState& self) const noexcept {
    return m_writer_active.load(std::memory_order_relaxed) == &self;
  }

  // Advisory check for destroy. Races can only make it report "busy" spuriously.
  [[nodiscard]] bool is_quiescent() const noexcept {
    if (m_writer_active.load(std::memory_order_relaxed) != nullptr || m_write_mutex.is_locked()) return false;
    if (any_indicator_busy()) return false;
    if (bias_compiled && runtime.m_bias_enabled && !no_visible_readers()) return false;
    return true;
  }

 private:
  [[nodiscard]] static std::size_t allocation_size() noexcept { return sizeof(RwLockCore) + Mutex::grant_bytes(); }

  // The row belongs to this thread and the counter has its own cache line, so this
  // is not a shared write. Atomic only because a sampling writer may read it.
  [[gnu::always_inline]] void count_read(BiasRow& row) const noexcept {
    auto& count = row.m_read_counts[m_bias_column];
    count.store(count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }

  void fix_profile(std::uint32_t percent) noexcept {
    const LoadProfile profile = percent >= heavy_enter_percent   ? LoadProfile::write_heavy
                                : percent >= mixed_enter_percent ? LoadProfile::mixed
                                                                 : LoadProfile::read_mostly;
    m_profile.store(profile, std::memory_order_relaxed);
  }

  // read_mostly: readers behind a leader park almost at once, and fast reads are on.
  // mixed:       readers spin for the next group, because one is coming soon.
  // write_heavy: as mixed, with fast reads off, because revoking them keeps paying.
  void apply_profile(LoadProfile profile) noexcept {
    const std::uint32_t patience = runtime.m_patience_pauses;
    m_group_patience.store(profile == LoadProfile::read_mostly ? patience / group_patience_divisor : patience,
                           std::memory_order_relaxed);
    m_bias_allowed.store(profile == LoadProfile::write_heavy ? 0 : 1, std::memory_order_relaxed);
  }

  // Runs with the write mutex held and the bias revoked. Columns are shared when a
  // process has many locks, which blends their counts: the estimate is approximate.
  [[gnu::noinline]] void sample_load() noexcept {
    const std::uint64_t now = read_clock();
    if (now - m_write_mutex.m_holder.m_sample_began < sample_interval_cycles) return;
    m_write_mutex.m_holder.m_sample_began = now;
    std::uint32_t reads = 0;
    const std::size_t rows = bias_table.high_water();
    for (std::size_t index = 0; index < rows; ++index) {
      reads += bias_table.row(index).m_read_counts[m_bias_column].load(std::memory_order_relaxed);
    }
    const std::uint64_t read_delta = reads - m_write_mutex.m_holder.m_sampled_reads;
    const std::uint64_t write_delta = m_write_mutex.m_holder.m_writes - m_write_mutex.m_holder.m_sampled_writes;
    m_write_mutex.m_holder.m_sampled_reads = reads;
    m_write_mutex.m_holder.m_sampled_writes = m_write_mutex.m_holder.m_writes;
    const std::uint64_t percent = write_delta * 100 / std::max<std::uint64_t>(read_delta + write_delta, 1);
    const LoadProfile current = m_profile.load(std::memory_order_relaxed);
    LoadProfile next = current;
    if (percent >= heavy_enter_percent) {
      next = LoadProfile::write_heavy;
    } else if (percent < mixed_leave_percent) {
      next = LoadProfile::read_mostly;
    } else if (percent < heavy_leave_percent && current == LoadProfile::write_heavy) {
      next = LoadProfile::mixed;
    } else if (percent >= mixed_enter_percent && current == LoadProfile::read_mostly) {
      next = LoadProfile::mixed;
    }
    // With clearly more threads than CPUs, waiters must get off the CPU whatever the
    // mix. The margin matters: one thread per CPU plus an idle main thread is not that.
    const auto cpus = static_cast<std::size_t>(runtime.m_cpu_count);
    if (bias_table.rows_in_use() > cpus + cpus / 4) next = LoadProfile::read_mostly;
    if (next != current) {
      m_profile.store(next, std::memory_order_relaxed);
      apply_profile(next);
    }
  }

  [[gnu::always_inline]] static void publish_fence() noexcept {
    if constexpr (asymmetric_compiled && !LATCH_TSAN) {
      if (runtime.m_asymmetric) [[likely]] {
        std::atomic_signal_fence(std::memory_order_seq_cst);  // writer runs membarrier()
        return;
      }
    }
    store_load_fence();
  }

  [[nodiscard]] NodeTicketLock& local_ticket() noexcept {
    return m_nodes[current_node()]->m_ticket;
  }

  void depart(ReaderIndicator& readers) noexcept {
    fetch_increment(readers.m_egress);
    if constexpr (parking_compiled) {
      // Clearing before waking closes the window where the writer has checked the
      // count but not yet slept: its futex_wait then fails on the changed flag.
      // Seq-cst: this load must not pass the increment above, or the writer can miss the
      // departure while this misses the flag, and the writer sleeps with no timeout.
      if (m_drain_parked.load(std::memory_order_seq_cst) != 0 &&
          m_drain_parked.exchange(0, std::memory_order_seq_cst) != 0) [[unlikely]] {
        futex_wake_all(m_drain_parked);
      }
    }
  }

  [[gnu::noinline]] int read_acquire_slow(ThreadState& self) noexcept {
    if (is_write_holder(self)) {
      if (runtime.m_warn) diagnostic("Warning: rdlock EDEADLK on %p\n", static_cast<void*>(this));
      return EDEADLK;
    }
    ReadMemo& memo = self.memo_for(this);
    if (++memo.m_count > 1) return 0;  // nested: must not wait for a pending writer
    if (self.m_bias_row != nullptr) count_read(*self.m_bias_row);

    ReaderIndicator& readers = m_nodes[memo.m_node]->m_readers;
    for (;;) {
      fetch_increment(readers.m_ingress);
      if (m_mode.load(std::memory_order_seq_cst) == LockMode::reading) break;

      // A writer is visible: readers defer. Retract, and gather behind a leader.
      depart(readers);
      if (fetch_increment(m_read_arrive) == 0) {
        m_write_mutex.lock(local_ticket());
        m_mode.store(LockMode::reading, std::memory_order_release);
        const std::uint32_t group = m_read_arrive.exchange(0, std::memory_order_seq_cst);
        LATCH_CLAIM(m_read_depart.load(std::memory_order_relaxed) == 0);
        m_read_depart.store(group, std::memory_order_release);
        if constexpr (parking_compiled) {
          store_load_fence();
          if (m_depart_parked.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            futex_wake_all(m_read_depart);
          }
        }
      }
      if (await_departure() == 1) {
        // Last of the group: it owes the release of the write mutex. Deferring that
        // to unlock keeps the window open for the group, and this thread need not
        // touch the indicator at all.
        memo.m_holds_write_mutex = true;
        break;
      }
      // The last of the group may have run ahead and let a writer in already, so
      // nothing is guaranteed here: try again from the top.
    }
    maybe_enable_bias();
    return 0;
  }

  // Takes one departure permit; returns how many were available before.
  std::uint32_t await_departure() noexcept {
    std::uint64_t spent = 0;
    const std::uint64_t budget = m_group_patience.load(std::memory_order_relaxed);
    for (;;) {
      std::uint32_t permits = m_read_depart.load(std::memory_order_acquire);
      if (permits == 0) {
        cpu_pause();
        if constexpr (parking_compiled) {
          if (runtime.m_patience >= 0 && ++spent >= budget) {
            m_depart_parked.fetch_add(1, std::memory_order_seq_cst);
            if (m_read_depart.load(std::memory_order_seq_cst) == 0) futex_wait(m_read_depart, 0);
            m_depart_parked.fetch_sub(1, std::memory_order_relaxed);
            spent = 0;
          }
        }
        continue;
      }
      if (m_read_depart.compare_exchange_weak(permits, permits - 1, std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
        return permits;
      }
    }
  }

  void maybe_enable_bias() noexcept {
    if constexpr (bias_compiled) {
      // We hold read permission, so no writer is past its drain. It will see this
      // store when it gets there, as our departure is ordered after it.
      if (runtime.m_bias_enabled && m_bias_allowed.load(std::memory_order_relaxed) != 0 &&
          m_bias.load(std::memory_order_relaxed) == 0 && read_clock() >= m_inhibit_until.load(std::memory_order_relaxed)) {
        m_bias.store(1, std::memory_order_release);
      }
    }
  }

  [[gnu::noinline]] int release_slow(ThreadState& self) noexcept {
    // A thread with write permission excludes all readers, so an active writer
    // means the caller is that writer.
    if (m_writer_active.load(std::memory_order_relaxed) != nullptr) {
      write_release();
      return 0;
    }
    ReadMemo* const memo = self.find_memo(this);
    if (memo == nullptr) return EPERM;
    LATCH_CLAIM(memo->m_count != 0);
    if (--memo->m_count != 0) return 0;
    const bool owes_mutex = memo->m_holds_write_mutex;
    const std::int32_t node = memo->m_node;
    memo->m_holds_write_mutex = false;
    memo->m_lock = nullptr;
    if (owes_mutex) {
      m_write_mutex.unlock();
    } else {
      depart(m_nodes[node]->m_readers);
    }
    return 0;
  }

  void write_release() noexcept {
    m_writer_active.store(nullptr, std::memory_order_relaxed);
    if constexpr (eager_reopen_compiled) {
      // Nobody queued: leaving the mode set would send the next reader through the
      // write mutex. A writer arriving just after this pays only a drain.
      if (!m_write_mutex.has_waiters()) {
        m_mode.store(LockMode::reading, std::memory_order_release);
      }
    }
    m_write_mutex.unlock();
  }

  [[nodiscard]] bool any_indicator_busy() const noexcept {
    for (int node = 0; node < runtime.m_node_count; ++node) {
      if (m_nodes[node]->m_readers.effective_count() != 0) return true;
    }
    return false;
  }

  void drain_readers() noexcept {
    for (int node = 0; node < runtime.m_node_count; ++node) drain(m_nodes[node]->m_readers);
  }

  // New arrivals retract, so waiting on a snapshot of ingress terminates. Only the
  // egress word is polled. Budget restarts when readers make progress.
  void drain(const ReaderIndicator& readers) noexcept {
    std::uint32_t in = readers.m_ingress.load(std::memory_order_acquire);
    std::uint32_t last_out = readers.m_egress.load(std::memory_order_acquire);
    const std::uint64_t budget = std::uint64_t{runtime.m_patience_pauses} * wait_patience_factor;
    std::uint64_t spent = 0;
    for (;;) {
      while (static_cast<std::int32_t>(in - last_out) > 0) {
        cpu_pause();
        const std::uint32_t out = readers.m_egress.load(std::memory_order_acquire);
        if (out != last_out) {
          last_out = out;
          spent = 0;
          continue;
        }
        if constexpr (parking_compiled) {
          if (runtime.m_patience >= 0 && ++spent >= budget) {
            m_drain_parked.store(1, std::memory_order_relaxed);
            store_load_fence();
            if (readers.effective_count() != 0) futex_wait(m_drain_parked, 1);
            m_drain_parked.store(0, std::memory_order_relaxed);
            last_out = readers.m_egress.load(std::memory_order_acquire);
            spent = 0;
          }
        }
      }
      const std::uint32_t verify = readers.m_ingress.load(std::memory_order_acquire);
      if (verify == in) return;
      in = verify;
    }
  }

  void close_bias_and_fence() noexcept {
    m_bias.store(0, std::memory_order_relaxed);
    heavy_fence();
  }

  void inhibit_bias(std::uint64_t began) noexcept {
    const std::uint64_t finished = read_clock();
    m_inhibit_until.store(finished + (finished - began) * runtime.m_bias_multiplier,
                          std::memory_order_relaxed);
  }

  // Revocation cost is charged back: fast reads stay off for a multiple of the time
  // revocation took, which bounds its share of writer time however often writers
  // come. A row added after we read the high-water mark belongs to a thread that
  // will see the cleared bias.
  [[gnu::noinline]] void revoke_bias() noexcept {
    const std::uint64_t began = read_clock();
    close_bias_and_fence();
    const std::size_t rows = bias_table.high_water();
    for (std::size_t index = 0; index < rows; ++index) {
      auto& slot = bias_table.row(index).m_slots[m_bias_column];
      std::uint32_t spent = 0;
      while (slot.load(std::memory_order_acquire) == this) {
        // No wake channel: the fast release is a single store with no check. Back
        // off to sleeping so a descheduled reader gets a CPU to finish on.
        if (++spent < runtime.m_patience_pauses || runtime.m_patience < 0) {
          cpu_pause();
        } else {
          short_sleep(50'000);
        }
      }
    }
    inhibit_bias(began);
  }

  [[nodiscard]] bool no_visible_readers() const noexcept {
    const std::size_t rows = bias_table.high_water();
    for (std::size_t index = 0; index < rows; ++index) {
      if (bias_table.row(index).m_slots[m_bias_column].load(std::memory_order_acquire) == this) return false;
    }
    return true;
  }

  [[nodiscard]] bool try_revoke_bias() noexcept {
    if (m_bias.load(std::memory_order_relaxed) == 0) return true;
    // A reader in plain view: fail without the fence, which interrupts every running
    // thread and would otherwise be paid on each retry of a timed acquisition.
    if (!no_visible_readers()) return false;
    const std::uint64_t began = read_clock();
    close_bias_and_fence();
    const bool clear = no_visible_readers();
    inhibit_bias(began);
    // A reader is still in its slot, so writers must keep scanning for it. Writers decide
    // whether to scan from the bias, and we still hold the write mutex.
    if (!clear) m_bias.store(1, std::memory_order_relaxed);
    return clear;
  }

  // Never written after construction: not invalidated by writer traffic.
  std::array<NodeState*, max_numa_nodes> m_nodes{};

  // Every reader reads this unit; writers write it. Readers only need it to be
  // current when a writer has been through, at which point they miss regardless.
  alignas(light_unit) std::atomic<const ThreadState*> m_writer_active{nullptr};
  std::atomic<LockMode> m_mode{LockMode::reading};
  std::atomic<std::uint32_t> m_bias{0};
  std::atomic<std::uint32_t> m_drain_parked{0};
  std::uint32_t m_bias_column = 0;
  std::atomic<std::uint64_t> m_inhibit_until{0};
  std::atomic<std::uint32_t> m_group_patience{0};
  std::atomic<std::uint32_t> m_bias_allowed{1};
  std::atomic<LoadProfile> m_profile{LoadProfile::read_mostly};
  bool m_adaptive = false;

  // Reader groups. Separate units: sharing one saves little and cost 10% at 1-5% writes.
  alignas(light_unit) std::atomic<std::uint32_t> m_read_arrive{0};
  alignas(light_unit) std::atomic<std::uint32_t> m_read_depart{0};
  std::atomic<std::uint32_t> m_depart_parked{0};

  // Must stay last: its grant slots follow it in memory.
  Mutex m_write_mutex;
};

// Native interface. Meets the SharedTimedMutex requirements, so std::unique_lock
// and std::shared_lock work. The object is one word; the state it refers to is
// type-stable, which makes destroying the lock right after an unlock safe.
template <Recursion recursion>
class BasicRwLock {
 public:
  // Adjusts itself to the observed mix of reads and writes.
  BasicRwLock() noexcept : m_core(pack(RwLockCore::create())) {}
  // Fixed configuration, no adjustment: RwLock lock{ExpectedWrites{50}};
  explicit BasicRwLock(ExpectedWrites expected) noexcept : m_core(pack(RwLockCore::create(expected))) {}
  ~BasicRwLock() { RwLockCore::dispose(core_ptr()); }
  BasicRwLock(const BasicRwLock&) = LATCH_DELETE("a lock has identity");
  BasicRwLock& operator=(const BasicRwLock&) = LATCH_DELETE("a lock has identity");

  // Throw std::system_error if this thread already holds the lock in a way that cannot
  // nest (resource_deadlock_would_occur), or at the limit on nesting
  // (resource_unavailable_try_again). Nothing is acquired, what the thread held is
  // unchanged, and a guard being constructed does not end up owning anything.
  void lock() {
    ThreadState& self = current_thread();
    if (nest_write(self)) return;
    const int status = core_ptr()->write_acquire(self);
    if (status != 0) [[unlikely]] {
      throw_refused(is_recursive && core_ptr()->is_write_holder(self) ? EAGAIN : status);
    }
  }
  void unlock() noexcept {
    if constexpr (is_recursive) {
      const std::uintptr_t word = m_core.load(std::memory_order_relaxed);
      if ((word >> depth_shift) != 0) {
        m_core.store(word - one_level, std::memory_order_relaxed);
      } else {
        (void)core_of(word)->release(current_thread());
      }
    } else {
      (void)m_core->release(current_thread());
    }
  }
  void lock_shared() {
    const int status = core_ptr()->read_acquire(current_thread());
    if (status != 0) [[unlikely]] throw_refused(status);
  }
  // As unlock(): the core releases whichever this thread holds.
  void unlock_shared() noexcept { unlock(); }
  [[nodiscard]] bool try_lock() noexcept {
    ThreadState& self = current_thread();
    return nest_write(self) || core_ptr()->try_write_acquire(self) == 0;
  }
  [[nodiscard]] bool try_lock_shared() noexcept { return core_ptr()->try_read_acquire(current_thread()) == 0; }

  template <class Clock, class Duration>
  [[nodiscard]] bool try_lock_until(const std::chrono::time_point<Clock, Duration>& deadline) noexcept {
    return poll_until(deadline, [this] { return try_lock(); },
                      [this] { return core_ptr()->held_by(current_thread()); });
  }
  template <class Clock, class Duration>
  [[nodiscard]] bool try_lock_shared_until(const std::chrono::time_point<Clock, Duration>& deadline) noexcept {
    return poll_until(deadline, [this] { return try_lock_shared(); },
                      [this] { return core_ptr()->is_write_holder(current_thread()); });
  }
  template <class Rep, class Period>
  [[nodiscard]] bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout) noexcept {
    return try_lock_until(deadline_after(timeout));
  }
  template <class Rep, class Period>
  [[nodiscard]] bool try_lock_shared_for(const std::chrono::duration<Rep, Period>& timeout) noexcept {
    return try_lock_shared_until(deadline_after(timeout));
  }

  [[nodiscard]] RwLockCore& core() noexcept { return *core_ptr(); }

 private:
  static constexpr bool is_recursive = recursion == Recursion::write;

  // With recursion, the top bits of the core pointer hold the levels beyond the first, so
  // the lock stays one word. Only the holder changes them. The word is atomic because
  // readers load the pointer bits meanwhile; those never change. User addresses fit: mmap
  // stays below 2^47 unless asked to go higher, and pack() checks.
  static_assert(!is_recursive || sizeof(std::uintptr_t) == 8);
  static constexpr int depth_shift = 54;
  static constexpr std::uint64_t one_level = std::uint64_t{1} << depth_shift;
  static_assert(max_write_nesting - 1 < (std::uint64_t{1} << (64 - depth_shift)));

  using Word = std::conditional_t<is_recursive, std::atomic<std::uintptr_t>, RwLockCore*>;

  [[nodiscard]] static auto pack(RwLockCore* core) noexcept {
    if constexpr (is_recursive) {
      const auto word = reinterpret_cast<std::uintptr_t>(core);
      if ((word >> depth_shift) != 0) [[unlikely]] {
        diagnostic("Error: lock at %p is too high in memory to hold a depth\n", static_cast<void*>(core));
        _exit(1);
      }
      return word;
    } else {
      return core;
    }
  }

  [[nodiscard, gnu::always_inline]] static RwLockCore* core_of(std::uintptr_t word) noexcept {
    return reinterpret_cast<RwLockCore*>(static_cast<std::uintptr_t>(word & (one_level - 1)));
  }

  [[nodiscard, gnu::always_inline]] RwLockCore* core_ptr() const noexcept {
    if constexpr (is_recursive) {
      return core_of(m_core.load(std::memory_order_relaxed));
    } else {
      return m_core;
    }
  }

  [[gnu::always_inline]] bool nest_write([[maybe_unused]] const ThreadState& self) noexcept {
    if constexpr (is_recursive) {
      const std::uintptr_t word = m_core.load(std::memory_order_relaxed);
      if (core_of(word)->is_write_holder(self) && (word >> depth_shift) != max_write_nesting - 1) [[likely]] {
        m_core.store(word + one_level, std::memory_order_relaxed);  // holding the lock is the first
        return true;
      }
    }
    return false;
  }

  [[noreturn, gnu::cold, gnu::noinline]] static void throw_refused(int status) {
#if defined(__cpp_exceptions)
    throw std::system_error(std::make_error_code(static_cast<std::errc>(status)));
#else
    // Built without exceptions, nothing can carry the error. As libstdc++ does here.
    diagnostic("Fatal: latch::RwLock refused: %s\n",
               status == EDEADLK ? "thread already holds this lock" : "too many nested locks");
    std::abort();
#endif
  }

  // now() + timeout overflows for a timeout such as hours::max(), which means to wait for
  // good. Compared as floating point, where neither side overflows and rounding keeps the
  // order, so the addition below is in range.
  template <class Rep, class Period>
  [[nodiscard]] static std::chrono::steady_clock::time_point deadline_after(
      const std::chrono::duration<Rep, Period>& timeout) noexcept {
    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    const Clock::time_point now = Clock::now();
    if (timeout <= std::chrono::duration<Rep, Period>::zero()) return now;
    if (Seconds(timeout) >= Seconds(Clock::time_point::max() - now)) return Clock::time_point::max();
    return now + std::chrono::ceil<Clock::duration>(timeout);
  }

  // Ticket queues cannot be abandoned, so timed acquisition polls the try path.
  // can_never_succeed: this thread's own hold is in the way, so waiting cannot help.
  template <class TimePoint>
  static bool poll_until(const TimePoint& deadline, std::predicate auto attempt,
                         std::predicate auto can_never_succeed) noexcept {
    for (long nap = 1'000;; nap = std::min(nap * 2, 1'000'000L)) {
      if (attempt()) return true;
      if (can_never_succeed() || TimePoint::clock::now() >= deadline) return false;
      short_sleep(nap);
    }
  }

  Word m_core;
};

}  // namespace latch
