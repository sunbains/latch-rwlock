// SPDX-License-Identifier: MIT OR Apache-2.0
// Process-wide state, per-thread state and allocation. Everything here tolerates running inside an LD_PRELOAD
// constructor, before the C++ runtime or the application has initialized.
#pragma once

#include <sched.h>
#include <linux/membarrier.h>
#include <linux/mempolicy.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <new>

#include "latch/diagnostic.hpp"
#include "latch/platform.hpp"
#include "latch/spin_lock.hpp"
#include "latch/topology.hpp"

namespace latch {

// Read-mostly configuration, sequestered from anything that gets written.
class alignas(coherence_unit) Runtime {
 public:
  const int* m_node_map = nullptr;
  int m_cpu_capacity = 0;
  int m_node_count = 1;         // after folding, in [1, max_numa_nodes]
  int m_cpu_count = 1;
  std::uint32_t m_grant_mask = 0;   // top-level grant slots - 1
  std::uint32_t m_grant_shift = 0;  // log2 of the top-level grant slots
  int m_run_bound = 100;        // RunBound: fairness limit on local hand-off
  int m_patience = 2000;        // Patience: nominal pauses before parking; <0 spins
  std::uint32_t m_admission_limit = 8;  // AdmissionLimit: queue depth per node; 0 disables
  int m_expected_writes = -1;    // ExpectedWrites: 0-100 fixes the profile; <0 is automatic
  std::uint32_t m_pause_q8 = 256;
  std::uint32_t m_patience_pauses = 0;
  std::uint32_t m_bias_multiplier = 9;
  bool m_bias_enabled = false;
  bool m_asymmetric = false;
  bool m_node_placement = false;
  bool m_warn = true;
  bool m_verbose = false;
};

LATCH_SHARED_STATE inline constinit Runtime runtime;

inline void ensure_runtime() noexcept;

// Run-time assertion, compiled out by default.
#define LATCH_CLAIM(condition)                                            \
  do {                                                                     \
    if constexpr (::latch::claims_compiled) {                             \
      if (!(condition))                                                    \
        ::latch::diagnostic("Claim failed '%s' at %s:%d\n", #condition,   \
                             __FILE__, __LINE__);                          \
    }                                                                      \
  } while (false)

// Convert a spin count expressed in "nominal" pauses (~10 cycles) into real pauses
// for this CPU, since the cost of PAUSE varies widely between x86 generations.
[[nodiscard]] inline std::uint32_t scaled_pauses(std::uint32_t nominal) noexcept {
  const std::uint64_t scaled = (std::uint64_t{nominal} * runtime.m_pause_q8) >> 8;
  return std::max<std::uint32_t>(1, static_cast<std::uint32_t>(scaled));
}

inline void pause_for(std::uint32_t nominal) noexcept { n_pause(scaled_pauses(nominal)); }

// NodeRestrictionMask and node folding are applied to the map once at start-up
// instead of on every call.
[[nodiscard]] inline int current_node() noexcept {
  const int cpu = current_cpu();
  return cpu < runtime.m_cpu_capacity ? runtime.m_node_map[cpu] : 0;
}

// ---- Type-stable block allocation --------------------------------------------
//
// mmap-backed and never returned to the OS. Two properties matter:
//  * no malloc, so a malloc that itself uses pthread_rwlock cannot recurse;
//  * type-stable memory, so an unlock may safely touch lock words after the store
//    that releases the lock, even if another thread destroys the lock right then.

enum class PoolKind { thread_state, read_memo, lock_core, node_state };

// node < 0 means no placement preference.
[[nodiscard]] inline void* pool_allocate(PoolKind kind, std::size_t size, int node = -1) noexcept;
inline void pool_release(PoolKind kind, void* block, int node = -1) noexcept;

// ---- Visible-reader table ----------------------------------------------------
//
// One row per thread. A reader publishes the lock it holds in its own row, so readers
// of even the same lock never share a cache line. Writers only have to look at one
// column (the lock's), in the rows in use. Read counts live in the same row, on their
// own unit, and are only ever read by a writer sampling the lock's load.

inline constexpr std::size_t bias_columns = 16;
inline constexpr std::size_t bias_row_capacity = 65536;

struct alignas(coherence_unit) BiasRow {
  std::atomic<const void*> m_slots[bias_columns];
  alignas(coherence_unit) std::atomic<std::uint32_t> m_read_counts[bias_columns];
};

class BiasTable {
 public:
  [[nodiscard]] BiasRow* acquire_row() noexcept;
  void release_row(BiasRow* row) noexcept;
  [[nodiscard]] std::size_t high_water() const noexcept {
    return m_high_water.load(std::memory_order_acquire);
  }
  [[nodiscard]] BiasRow& row(std::size_t index) const noexcept { return m_rows[index]; }
  // Rows held by live threads. Racy, and only used as an estimate.
  [[nodiscard]] std::size_t rows_in_use() const noexcept {
    return m_high_water.load(std::memory_order_relaxed) - m_free_count.load(std::memory_order_relaxed);
  }
  void initialize() noexcept;

 private:
  BiasRow* m_rows = nullptr;
  std::uint32_t* m_free_rows = nullptr;
  std::atomic<std::size_t> m_free_count{0};
  std::atomic<std::size_t> m_high_water{0};
  SpinLock m_lock;
};

LATCH_SHARED_STATE inline constinit BiasTable bias_table;

// Writer half of the asymmetric Dekker protocol with the reader's unfenced publish.
inline void heavy_fence() noexcept;

// ---- Per-thread state --------------------------------------------------------

// Tracks a read permission obtained through the reader indicators.
struct ReadMemo {
  const void* m_lock = nullptr;
  ReadMemo* m_next = nullptr;
  std::uint32_t m_count = 0;       // nesting depth
  std::int32_t m_node = 0;         // the indicator we incremented
  bool m_holds_write_mutex = false;  // last reader of a group
  bool m_from_pool = false;
};

inline constexpr std::uint32_t thread_type_tag = 0x7CE55A1DU;

class alignas(coherence_unit) ThreadState {
 public:
  ThreadState() noexcept;

  // Hot: touched by every read acquire and release.
  BiasRow* m_bias_row = nullptr;
  ReadMemo* m_read_list = nullptr;
  std::uint32_t m_bias_nesting[bias_columns] = {};

  // A few memos live inline: no allocation, and no false sharing with another
  // thread's memos.
  ReadMemo m_inline_memos[4];

  std::uint32_t m_type_tag = thread_type_tag;
  pid_t m_lwp_id = 0;

  // Find the memo for lock, or claim a free one. The list is short and thread-private, so a linear scan is the right structure.
  [[nodiscard]] ReadMemo& memo_for(const void* lock) noexcept;
  [[nodiscard]] ReadMemo* find_memo(const void* lock) const noexcept {
    for (ReadMemo* memo = m_read_list; memo != nullptr; memo = memo->m_next) {
      if (memo->m_lock == lock) return memo;
    }
    return nullptr;
  }
};

LATCH_SHARED_STATE [[gnu::tls_model("initial-exec")]] inline constinit thread_local ThreadState*
    tls_thread_state = nullptr;

[[gnu::cold]] inline ThreadState& register_current_thread() noexcept;

// TLS load on the fast path, lazy registration otherwise.
[[gnu::always_inline]] inline ThreadState& current_thread() noexcept {
  ThreadState* self = tls_thread_state;
  if (self != nullptr) [[likely]] {
    LATCH_CLAIM(self->m_type_tag == thread_type_tag);
    return *self;
  }
  return register_current_thread();
}

// ---- Implementation -----------------------------------------------------------

namespace detail {

inline constexpr int runtime_uninitialized = 0;
inline constexpr int runtime_initializing = 1;
inline constexpr int runtime_ready = 2;

LATCH_SHARED_STATE inline constinit std::atomic<int> runtime_state{runtime_uninitialized};
LATCH_SHARED_STATE inline constinit pthread_key_t thread_key{};
LATCH_SHARED_STATE inline constinit const int* kernel_node_ids = nullptr;

// CPUs this process may run on, which a cpuset or affinity mask can make far fewer
// than the machine has. Decides when threads outnumber CPUs. Asks about the main thread,
// not this one: the first thread to use a lock may be pinned, unlike the process.
[[nodiscard]] inline int usable_cpu_count(int discovered) noexcept {
  cpu_set_t allowed;
  if (sched_getaffinity(getpid(), sizeof(allowed), &allowed) != 0) return discovered;
  const int count = CPU_COUNT(&allowed);
  return count > 0 ? std::min(count, discovered) : discovered;
}

[[nodiscard]] inline int eval_key(const char* name, int fallback) noexcept {
  const char* text = std::getenv(name);
  if (text == nullptr) return fallback;
  const int value = static_cast<int>(std::strtol(text, nullptr, 0));
  if (runtime.m_verbose && value != fallback) {
    diagnostic("Override: %s %d->%d\n", name, fallback, value);
  }
  return value;
}

// ---- Pools -------------------------------------------------------------------

class BlockPool {
 public:
  [[nodiscard]] void* allocate(std::size_t size, int node) noexcept {
    m_lock.lock();
    void* block = m_free;
    if (block != nullptr) {
      m_free = *static_cast<void**>(block);
    } else {
      block = bump(size, node);
    }
    m_lock.unlock();
    return block;
  }

  void release(void* block) noexcept {
    m_lock.lock();
    *static_cast<void**>(block) = m_free;
    m_free = block;
    m_lock.unlock();
  }

  SpinLock m_lock;

 private:
  [[nodiscard]] void* bump(std::size_t size, int node) noexcept {
    const std::size_t rounded = (size + coherence_unit - 1) & ~(coherence_unit - 1);
    if (static_cast<std::size_t>(m_end - m_cursor) < rounded) {
      const std::size_t chunk = std::max<std::size_t>(1024 * 1024, rounded * 16);
      void* region = mmap(nullptr, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (region == MAP_FAILED) {
        diagnostic("Error: mmap(%zu) failed\n", chunk);
        _exit(1);
      }
      if (node >= 0 && runtime.m_node_placement) place_on_node(region, chunk, node);
      m_cursor = static_cast<char*>(region);
      m_end = m_cursor + chunk;
    }
    void* block = m_cursor;
    m_cursor += rounded;
    return block;
  }

  // Home the chunk's pages on the node that will write to it. MPOL_PREFERRED so
  // that a full node degrades to remote memory instead of failing the fault.
  static void place_on_node(void* region, std::size_t size, int node) noexcept {
    constexpr std::size_t mask_bits = 1024;
    unsigned long mask[mask_bits / (8 * sizeof(unsigned long))] = {};
    const auto kernel_node = static_cast<std::size_t>(kernel_node_ids[node]);
    if (kernel_node >= mask_bits) return;
    mask[kernel_node / (8 * sizeof(unsigned long))] |= 1UL << (kernel_node % (8 * sizeof(unsigned long)));
    syscall(SYS_mbind, region, size, MPOL_PREFERRED, mask, mask_bits, 0);
  }

  void* m_free = nullptr;
  char* m_cursor = nullptr;
  char* m_end = nullptr;
};

LATCH_SHARED_STATE inline constinit BlockPool shared_pools[3];
LATCH_SHARED_STATE inline constinit BlockPool node_pools[max_numa_nodes];

[[nodiscard]] inline BlockPool& pool_for(PoolKind kind, int node) noexcept {
  if (kind == PoolKind::node_state) return node_pools[node < 0 ? 0 : node];
  return shared_pools[static_cast<int>(kind)];
}

// ---- Thread lifecycle --------------------------------------------------------

inline void reclaim_thread(ThreadState* thread) noexcept {
  if (tls_thread_state == thread) tls_thread_state = nullptr;

  ReadMemo* memo = thread->m_read_list;
  thread->m_read_list = nullptr;
  while (memo != nullptr) {
    ReadMemo* const next = memo->m_next;
    if (memo->m_from_pool) pool_release(PoolKind::read_memo, memo);
    memo = next;
  }

  if (BiasRow* row = thread->m_bias_row) bias_table.release_row(row);
  thread->m_type_tag = 0;
  pool_release(PoolKind::thread_state, thread);
}

inline void thread_exit_callback(void* value) noexcept {
  if (value != nullptr) reclaim_thread(static_cast<ThreadState*>(value));
}

[[nodiscard]] inline std::uint32_t measure_pause_q8() noexcept {
  if constexpr (!calibrate_pause_compiled) return 256;
  constexpr std::uint32_t nominal_cycles = 10;
  constexpr int samples = 5;
  constexpr std::uint32_t pauses_per_sample = 64;
  std::uint64_t best = ~std::uint64_t{0};
  for (int sample = 0; sample < samples; ++sample) {
    const std::uint64_t begin = read_clock();
    n_pause(pauses_per_sample);
    best = std::min(best, read_clock() - begin);  // minimum rejects interrupts
  }
  const std::uint64_t cycles = std::max<std::uint64_t>(best / pauses_per_sample, 1);
  // Never above 1:1. A very cheap PAUSE keeps the nominal count.
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(nominal_cycles * 256 / cycles, 256));
}

inline void initialize_runtime() noexcept {
  runtime.m_verbose = eval_key("VERBOSE", 0) != 0;
  runtime.m_warn = eval_key("Warn", 1) != 0 || runtime.m_verbose;
  runtime.m_run_bound = eval_key("RunBound", runtime.m_run_bound);
  runtime.m_patience = eval_key("Patience", runtime.m_patience);
  runtime.m_bias_multiplier = static_cast<std::uint32_t>(eval_key("BiasMultiplier", 9));
  runtime.m_admission_limit = static_cast<std::uint32_t>(eval_key("AdmissionLimit", 8));
  if (runtime.m_patience < 0) runtime.m_admission_limit = 0;  // pure spinning requested
  runtime.m_expected_writes = eval_key("ExpectedWrites", -1);
  const int restriction_mask = eval_key("NodeRestrictionMask", -1);

  const Topology topology = Topology::query(runtime.m_verbose);
  // More nodes than arrays: share indices, keeping most of the benefit.
  for (int cpu = 0; cpu < topology.m_cpu_capacity; ++cpu) {
    topology.m_node_map[cpu] = (topology.m_node_map[cpu] & restriction_mask) % max_numa_nodes;
  }
  const int folded = restriction_mask == -1 ? std::min(topology.m_node_count, max_numa_nodes)
                                            : (restriction_mask & (max_numa_nodes - 1)) + 1;
  runtime.m_node_map = topology.m_node_map;
  runtime.m_cpu_capacity = topology.m_cpu_capacity;
  runtime.m_cpu_count = usable_cpu_count(topology.m_cpu_count);
  runtime.m_node_count = folded;
  // One top-level waiter per node at most, so one slot per node gives each its own.
  const std::uint32_t grant_slots = std::bit_ceil(static_cast<std::uint32_t>(folded));
  runtime.m_grant_mask = grant_slots - 1;
  runtime.m_grant_shift = static_cast<std::uint32_t>(std::countr_zero(grant_slots));
  kernel_node_ids = topology.m_physical_node;
  runtime.m_node_placement = eval_key("NodePlacement", 0) != 0 && folded > 1;

  runtime.m_pause_q8 = measure_pause_q8();
  runtime.m_patience_pauses =
      runtime.m_patience < 0 ? 0 : scaled_pauses(static_cast<std::uint32_t>(runtime.m_patience));

  bool bias = bias_compiled && eval_key("Bias", 1) != 0;
  bool asymmetric = false;
  if (bias) {
    bias_table.initialize();
    if constexpr (asymmetric_compiled && !LATCH_TSAN) {
      asymmetric = eval_key("BiasAsymmetric", 1) != 0 &&
                   syscall(SYS_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0) == 0;
    }
  }
  runtime.m_asymmetric = asymmetric;
  runtime.m_bias_enabled = bias;

  if (pthread_key_create(&thread_key, thread_exit_callback) != 0) {
    diagnostic("Error: pthread_key_create failed\n");
    _exit(1);
  }
  if (runtime.m_verbose) {
    diagnostic("latch: nodes=%d pause_q8=%u patience=%u bias=%d asymmetric=%d placement=%d\n",
               runtime.m_node_count, runtime.m_pause_q8, runtime.m_patience_pauses, bias,
               asymmetric, runtime.m_node_placement);
  }
}

}  // namespace detail

inline void ensure_runtime() noexcept {
  int expected = detail::runtime_uninitialized;
  if (detail::runtime_state.load(std::memory_order_acquire) == detail::runtime_ready) return;
  if (!detail::runtime_state.compare_exchange_strong(expected, detail::runtime_initializing,
                                             std::memory_order_acquire)) {
    while (detail::runtime_state.load(std::memory_order_acquire) != detail::runtime_ready) cpu_pause();
    return;
  }
  detail::initialize_runtime();
  detail::runtime_state.store(detail::runtime_ready, std::memory_order_release);
}

inline void* pool_allocate(PoolKind kind, std::size_t size, int node) noexcept {
  return detail::pool_for(kind, node).allocate(size, node);
}

inline void pool_release(PoolKind kind, void* block, int node) noexcept {
  detail::pool_for(kind, node).release(block);
}

// ---- Bias table ----------------------------------------------------------------

inline void BiasTable::initialize() noexcept {
  // Reserved lazily by the kernel: untouched rows consume no memory.
  const std::size_t row_bytes = sizeof(BiasRow) * bias_row_capacity;
  const std::size_t free_bytes = sizeof(std::uint32_t) * bias_row_capacity;
  void* rows = mmap(nullptr, row_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  void* free_rows = mmap(nullptr, free_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (rows == MAP_FAILED || free_rows == MAP_FAILED) {
    diagnostic("Error: could not reserve the reader table\n");
    _exit(1);
  }
  m_rows = static_cast<BiasRow*>(rows);
  m_free_rows = static_cast<std::uint32_t*>(free_rows);
}

inline BiasRow* BiasTable::acquire_row() noexcept {
  BiasRow* row = nullptr;
  m_lock.lock();
  if (const std::size_t free_count = m_free_count.load(std::memory_order_relaxed); free_count != 0) {
    row = &m_rows[m_free_rows[free_count - 1]];
    m_free_count.store(free_count - 1, std::memory_order_relaxed);
  } else if (const std::size_t next = m_high_water.load(std::memory_order_relaxed);
             next < bias_row_capacity) {
    row = &m_rows[next];
    // Seq-cst pairs with revoke's read of the high-water mark; see RwLockCore.
    m_high_water.store(next + 1, std::memory_order_seq_cst);
  }
  m_lock.unlock();
  return row;  // nullptr: too many threads, this one just uses the slow path
}

inline void BiasTable::release_row(BiasRow* row) noexcept {
  m_lock.lock();
  const std::size_t free_count = m_free_count.load(std::memory_order_relaxed);
  m_free_rows[free_count] = static_cast<std::uint32_t>(row - m_rows);
  m_free_count.store(free_count + 1, std::memory_order_relaxed);
  m_lock.unlock();
}

inline void heavy_fence() noexcept {
  if (runtime.m_asymmetric) {
    if (syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0) == 0) [[likely]] return;
    diagnostic("Error: membarrier failed after successful registration\n");
    _exit(1);
  }
  store_load_fence();
}

// ---- Threads ----------------------------------------------------------------

inline ThreadState::ThreadState() noexcept {
  for (std::size_t index = 0; index + 1 < std::size(m_inline_memos); ++index) {
    m_inline_memos[index].m_next = &m_inline_memos[index + 1];
  }
  m_read_list = &m_inline_memos[0];
}

inline ReadMemo& ThreadState::memo_for(const void* lock) noexcept {
  ReadMemo* first_free = nullptr;
  for (ReadMemo* memo = m_read_list; memo != nullptr; memo = memo->m_next) {
    if (memo->m_lock == lock) return *memo;  // nested
    if (memo->m_lock == nullptr && first_free == nullptr) first_free = memo;
  }
  if (first_free == nullptr) {
    first_free = new (pool_allocate(PoolKind::read_memo, sizeof(ReadMemo))) ReadMemo{};
    first_free->m_from_pool = true;
    first_free->m_next = m_read_list;
    m_read_list = first_free;
  }
  first_free->m_node = current_node();
  first_free->m_lock = lock;
  first_free->m_count = 0;
  return *first_free;
}

inline ThreadState& register_current_thread() noexcept {
  ensure_runtime();
  auto* self = new (pool_allocate(PoolKind::thread_state, sizeof(ThreadState))) ThreadState{};
  self->m_lwp_id = static_cast<pid_t>(syscall(SYS_gettid));
  if (runtime.m_bias_enabled) self->m_bias_row = bias_table.acquire_row();
  // TLS first: pthread_setspecific may allocate, and that must not recurse here.
  tls_thread_state = self;
  pthread_setspecific(detail::thread_key, self);  // gives us the thread-exit callback
  if (runtime.m_verbose) {
    diagnostic("Thread %d @ %p node=%d\n", self->m_lwp_id, static_cast<void*>(self), current_node());
  }
  return *self;
}

}  // namespace latch
