// SPDX-License-Identifier: MIT OR Apache-2.0
// Synthetic reader-writer lock microbenchmark and stress test with lightweight
// integrity checks. Fixed measurement interval; reports aggregate work completed
// and per-thread fairness.
//
//   rw_bench [Key=Value ...] +<Threads>@<WritePct> [+<Threads>@<WritePct> ...]
//
// Lock=pthread (default) calls pthread_rwlock_*, so the same binary measures any
// LD_PRELOAD implementation. Lock=native uses latch::RwLock directly, Lock=recursive uses
// latch::RecursiveRwLock, and Lock=barging uses latch::BargingRwLock.
#include <alloca.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <print>
#include <string_view>
#include <vector>

#include "latch/latch.hpp"

namespace {

using latch::coherence_unit;
using latch::marsaglia_next;
using latch::mix_hash;

inline constexpr int fixed_base = 1024;
inline constexpr int fixed_mask = fixed_base - 1;
inline constexpr int last_write_shared = -1;

[[noreturn]] void fail(std::string_view what, int a = 0, int b = 0) {
  std::println("ERROR: {} {:X}:{:X}", what, a, b);
  std::exit(1);
}

class PthreadLock {
 public:
  PthreadLock() noexcept { pthread_rwlock_init(&m_lock, nullptr); }
  void lock() noexcept { check(pthread_rwlock_wrlock(&m_lock)); }
  void lock_shared() noexcept { check(pthread_rwlock_rdlock(&m_lock)); }
  void unlock() noexcept { check(pthread_rwlock_unlock(&m_lock)); }
  void unlock_shared() noexcept { check(pthread_rwlock_unlock(&m_lock)); }

 private:
  static void check(int status) noexcept {
    if (status != 0) [[unlikely]] fail("primitive failed", status);
  }
  pthread_rwlock_t m_lock;
};

template <class Lock>
class alignas(coherence_unit) RwNode {
 public:
  [[nodiscard]] static RwNode* create(int length) {
    if (!std::has_single_bit(static_cast<unsigned>(length))) fail("ALength must be a power of two");
    const std::size_t size = sizeof(RwNode) + sizeof(int) * (static_cast<std::size_t>(length) + 1) + coherence_unit;
    void* space = std::aligned_alloc(coherence_unit, (size + coherence_unit - 1) & ~(coherence_unit - 1));
    std::memset(space, 0, size);
    auto* node = new (space) RwNode;
    node->m_length = length;
    return node;
  }

  [[nodiscard]] volatile int* data() noexcept { return reinterpret_cast<volatile int*>(this + 1); }

  // Written once.
  int m_length = 0;

  alignas(coherence_unit) Lock m_lock;

  alignas(coherence_unit) std::atomic<int> m_last_access_type{0};  // read vs write
  std::atomic<int> m_last_write_node{0};  // tracks lock migration
  volatile int m_write_seq = 0;           // similar to a seqlock
};

struct ThreadBlock {
  int m_write_pct = 0;
  int m_rcs_length = 0;
  int m_rcs_delay = 0;
  int m_wcs_length = 0;
  int m_wcs_delay = 0;
  int m_ncs_length = 0;
  int m_ncs_delay = 0;
  int m_idelay = 0;
  int m_imask = 0;
  std::atomic<int> m_status{0};

  ThreadBlock() = default;
  ThreadBlock(const ThreadBlock& other) noexcept
      : m_write_pct(other.m_write_pct), m_rcs_length(other.m_rcs_length), m_rcs_delay(other.m_rcs_delay),
        m_wcs_length(other.m_wcs_length), m_wcs_delay(other.m_wcs_delay), m_ncs_length(other.m_ncs_length),
        m_ncs_delay(other.m_ncs_delay), m_idelay(other.m_idelay), m_imask(other.m_imask) {}
  ThreadBlock& operator=(const ThreadBlock& other) noexcept {
    if (this != &other) {
      this->~ThreadBlock();
      new (this) ThreadBlock(other);
    }
    return *this;
  }
};

enum class LockKind { pthread, native, recursive, barging };

struct Options {
  int m_verbose = 0;
  int m_instances = 1;
  int m_alength = 16;
  int m_duration = 10'000;
  int m_rdelay = 0;
  int m_vstress = 0;
  int m_barrier = 12;
  int m_bind = 0;
  const char* m_comment = "";
  LockKind m_kind = LockKind::pthread;
};

// Thread-to-CPU assignment for Bind=1: balance over nodes, then geographic order.
[[nodiscard]] std::vector<int> build_placement() {
  const latch::Topology topology = latch::Topology::query(false);
  std::vector<std::vector<std::pair<int, int>>> by_node(static_cast<std::size_t>(topology.m_node_count));
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};
  for (int cpu = 0; cpu < std::min(topology.m_cpu_capacity - 1, CPU_SETSIZE); ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;  // offline or outside this process's cpuset
    by_node[static_cast<std::size_t>(topology.m_node_map[cpu])].push_back({topology.m_geo_map[cpu], cpu});
  }
  std::vector<int> placement;
  for (auto& cpus : by_node) std::ranges::sort(cpus);
  for (std::size_t rank = 0;; ++rank) {
    bool any = false;
    for (const auto& cpus : by_node) {
      if (rank < cpus.size()) {
        placement.push_back(cpus[rank].second);
        any = true;
      }
    }
    if (!any) return placement;
  }
}

template <class Lock>
class RwBench {
 public:
  RwBench(const Options& options, std::vector<ThreadBlock>& threads) noexcept
      : m_options(options), m_threads(threads), m_thread_count(static_cast<int>(threads.size())) {}

  int run(const char* executable);

 private:
  struct WorkerStart {
    RwBench* m_bench;
    int m_id;
  };

  [[nodiscard]] static int to_node(int cpu) noexcept {
    return cpu < latch::runtime.m_cpu_capacity ? latch::runtime.m_node_map[cpu] : 0;
  }

  // Implement stack coloring via alloca, consuming the result so it is not elided.
  static void* thunk(void* raw) {
    constinit static std::atomic<int> color{0};
    void* const colored = alloca(64 * (color.fetch_add(1, std::memory_order_relaxed) & 1));
    if (colored == reinterpret_cast<void*>(0xDD)) std::print("!");
    const WorkerStart start = *static_cast<WorkerStart*>(raw);
    delete static_cast<WorkerStart*>(raw);
    start.m_bench->worker(start.m_id);
    return nullptr;
  }

  static void launch(RwBench* bench, int id) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_t thread{};
    if (const int status = pthread_create(&thread, &attr, thunk, new WorkerStart{bench, id}); status != 0) {
      fail("pthread_create", status);
    }
    pthread_attr_destroy(&attr);
  }

  static void* change_concurrency(void* raw) {
    auto* bench = static_cast<RwBench*>(raw);
    for (;;) {
      poll(nullptr, 0, bench->m_options.m_vstress);
      bench->m_bar_halt.store(1, std::memory_order_relaxed);
    }
  }

  void worker(int id);
  [[nodiscard]] int arrive_filling() noexcept;
  [[nodiscard]] int arrive_ticket() noexcept;

  const Options& m_options;
  std::vector<ThreadBlock>& m_threads;
  const int m_thread_count;

  std::vector<int> m_placement;
  std::vector<int> m_shuffle_map;
  std::vector<std::int64_t> m_its_completed;
  RwNode<Lock>** m_table = nullptr;  // sequestered: read on every operation

  alignas(coherence_unit) std::atomic<int> m_halt{0};
  std::atomic<int> m_start{0};
  std::atomic<int> m_bar_halt{0};
  alignas(coherence_unit) std::atomic<int> m_halted{0};
  std::atomic<int> m_ready{0};

  pthread_mutex_t m_gate = PTHREAD_MUTEX_INITIALIZER;
  std::int64_t m_lock_migrations_read = 0;
  std::int64_t m_lock_migrations_write = 0;
  std::int64_t m_total_raw = 0;  // write->read transitions
  std::int64_t m_total_reads = 0;
  std::int64_t m_total_writes = 0;
  std::int64_t m_its_by_node[latch::max_numa_nodes] = {};
  int m_node_distribution[latch::max_numa_nodes] = {};

  // Barrier state for VStress: quiesce all threads, then release a random subset.
  // Moving concurrency from M to 0 to N, rather than M to N, makes the lock change
  // between contended and uncontended often, which shakes out exclusion and
  // progress bugs that never show in steady state.
  alignas(coherence_unit) std::atomic<int> m_bar_arrived{0};
  std::atomic<int> m_bar_level{0};
  std::atomic<int> m_bar_draining{0};
  std::atomic<int> m_bar_ticket{0};
  std::atomic<int> m_bar_grant{0};
  std::atomic<int> m_bar_gate{0};
  std::uint32_t m_bar_random = 0;
};

// Distinct non-overlapping filling and draining phases.
template <class Lock>
int RwBench<Lock>::arrive_filling() noexcept {
  for (;;) {
    int arrived = m_bar_arrived.load(std::memory_order_acquire);
    if (arrived == -1) {  // frozen: busy-wait
      if (m_halt.load(std::memory_order_relaxed) != 0) return -1;
      latch::cpu_pause();
      continue;
    }
    if (arrived + 1 == m_thread_count) {
      // All threads captured: freeze arrivals and choose the next concurrency level.
      if (!m_bar_arrived.compare_exchange_strong(arrived, -1)) continue;
      m_bar_random = marsaglia_next(m_bar_random);
      const int level = static_cast<int>((mix_hash(m_bar_random >> 4) & 0x7FFFFFFFU) %
                                         static_cast<std::uint32_t>(m_thread_count)) + 1;
      if (m_options.m_verbose != 0) std::print("\nCL{} ", level);
      m_bar_halt.store(0, std::memory_order_relaxed);
      m_bar_level.store(level, std::memory_order_relaxed);
      m_bar_draining.store(level, std::memory_order_release);
      break;
    }
    if (m_bar_arrived.compare_exchange_strong(arrived, arrived + 1)) break;
  }
  for (;;) {
    int draining = m_bar_draining.load(std::memory_order_acquire);
    if (draining == 0) {
      if (m_halt.load(std::memory_order_relaxed) != 0) return -1;
      latch::cpu_pause();
      continue;
    }
    if (m_bar_draining.compare_exchange_strong(draining, draining - 1)) {
      if (draining == 1) {  // last of the group departs: unfreeze
        m_bar_arrived.store(m_thread_count - m_bar_level.load(std::memory_order_relaxed),
                            std::memory_order_release);
      }
      return 0;
    }
  }
}

// Lead thread waits inside a ticket-lock critical section for a quorum; followers
// wait at its entry.
template <class Lock>
int RwBench<Lock>::arrive_ticket() noexcept {
  const int ticket = m_bar_ticket.fetch_add(1);
  while (m_bar_grant.load(std::memory_order_acquire) != ticket) {
    latch::cpu_pause();
    if (m_halt.load(std::memory_order_relaxed) != 0) return -1;
  }
  if (m_bar_gate.load(std::memory_order_relaxed) == 0) {
    while (m_bar_ticket.load(std::memory_order_acquire) - ticket < m_thread_count) {
      latch::cpu_pause();
      if (m_halt.load(std::memory_order_relaxed) != 0) return -1;
    }
    m_bar_random = marsaglia_next(m_bar_random);
    m_bar_gate.store(static_cast<int>((mix_hash(m_bar_random >> 4) & 0x7FFFFFFFU) %
                                      static_cast<std::uint32_t>(m_thread_count)) + 1,
                     std::memory_order_relaxed);
    m_bar_halt.store(0, std::memory_order_relaxed);
  }
  m_bar_gate.fetch_sub(1, std::memory_order_relaxed);
  m_bar_grant.fetch_add(1, std::memory_order_release);
  return 0;
}

template <class Lock>
void RwBench<Lock>::worker(int id) {
  if (m_options.m_bind != 0 && !m_placement.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(m_placement[static_cast<std::size_t>(id) % m_placement.size()], &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) fail("could not bind thread", id);
  }

  RwNode<Lock>* const private_node = RwNode<Lock>::create(m_options.m_alength);

  // The shuffle prevents relationships between birth order, placement by the
  // scheduler and thread role from skewing results on NUMA systems.
  ThreadBlock& block = m_threads[static_cast<std::size_t>(m_shuffle_map[static_cast<std::size_t>(id)])];
  if (block.m_status.exchange(1) != 0) fail("thread slot in use", id);

  // Fixed point avoids a division in the main loop.
  const int write_probability = (block.m_write_pct * fixed_base) / 100;
  const int wcs_length = block.m_wcs_length;
  const int wcs_delay = block.m_wcs_delay;
  const int rcs_length = block.m_rcs_length;
  const int rcs_delay = block.m_rcs_delay;
  const int ncs_length = block.m_ncs_length;
  const int ncs_delay = block.m_ncs_delay;
  const int imask = block.m_imask;
  const int idelay = block.m_idelay;
  const int instances = m_options.m_instances;
  const int index_mask = m_options.m_alength - 1;
  const int rdelay = m_options.m_rdelay;

  m_ready.fetch_add(1);
  while (m_start.load(std::memory_order_acquire) == 0) latch::cpu_pause();

  int node_id = to_node(sched_getcpu());
  std::int64_t its = 0;
  std::uint32_t rv = static_cast<std::uint32_t>(id) + 0xD1CE;
  int writes = 0;
  int migrations_write = 0;
  int migrations_read = 0;
  int raw = 0;

  while (m_halt.load(std::memory_order_relaxed) == 0) {
    ++its;
    if ((its & 0xFFF) == 0) node_id = to_node(sched_getcpu());  // rare migration

    if (m_bar_halt.load(std::memory_order_relaxed) != 0) {
      (void)(m_options.m_barrier == 21 ? arrive_ticket() : arrive_filling());
      if (m_halt.load(std::memory_order_relaxed) != 0) break;
    }

    rv = marsaglia_next(rv);
    RwNode<Lock>* const node = m_table[rv & static_cast<std::uint32_t>(instances - 1)];
    volatile int* const data = node->data();

    rv = marsaglia_next(rv);
    if (static_cast<int>(rv & fixed_mask) < write_probability) {
      ++writes;
      if (wcs_length >= 0) {
        node->m_lock.lock();
        const int sequence = node->m_write_seq;
        node->m_write_seq = sequence + 1;
        if ((sequence & 1) != 0) fail("W-W exclusion failure", sequence);
        if (node->m_last_write_node.load(std::memory_order_relaxed) != node_id) {
          node->m_last_write_node.store(node_id, std::memory_order_relaxed);
          ++migrations_write;
        }
        node->m_last_access_type.store(1, std::memory_order_relaxed);
        const int term = id + 0x10;
        for (int step = 0; step < wcs_length; ++step) {
          rv = marsaglia_next(rv);
          const int first = static_cast<int>(rv) & index_mask;
          rv = marsaglia_next(rv);
          const int second = static_cast<int>(rv) & index_mask;
          data[first] = data[first] + term;  // conservation of the sum of the array
          data[second] = data[second] - term;
        }
        for (int delay = 0; delay < wcs_delay; ++delay) rv = marsaglia_next(rv);
        node->m_write_seq = node->m_write_seq + 1;
        node->m_lock.unlock();
      }
    } else if (rcs_length >= 0) {
      node->m_lock.lock_shared();
      const int before = node->m_write_seq;
      const int last_writer = node->m_last_write_node.load(std::memory_order_relaxed);
      if (last_writer != node_id && last_writer != last_write_shared) {
        // Local read after remote write. The exchange makes only the lead reader
        // count it, so read migrations are not overstated by the race.
        if (node->m_last_write_node.exchange(last_write_shared) != last_write_shared) ++migrations_read;
      }
      if (node->m_last_access_type.load(std::memory_order_relaxed) != 0 &&
          node->m_last_access_type.exchange(0) != 0) {
        ++raw;
      }
      int sum = 0;
      for (int step = 0; step < rcs_length; ++step) {
        rv = marsaglia_next(rv);
        const int first = static_cast<int>(rv) & index_mask;
        rv = marsaglia_next(rv);
        const int second = static_cast<int>(rv) & index_mask;
        sum += data[first];
        sum += data[second];
      }
      for (int delay = 0; delay < rcs_delay; ++delay) rv = marsaglia_next(rv);
      const int after = node->m_write_seq;
      node->m_lock.unlock_shared();
      if (sum == 0xDD) rv = marsaglia_next(rv);  // keep the reads above alive
      if (((before ^ after) | (before & 1)) != 0) fail("R-W exclusion failure", before, after);
    }

    if (ncs_length != 0) {
      // Non-critical work: the write body on thread-private data, fully scalable.
      rv = marsaglia_next(rv);
      volatile int* const private_data = private_node->data();
      const int sequence = private_node->m_write_seq;
      private_node->m_write_seq = sequence + 1;
      if ((sequence & 1) != 0) fail("private exclusion failure", sequence);
      const int term = id + 0x10;
      for (int step = 0; step < ncs_length; ++step) {
        rv = marsaglia_next(rv);
        const int first = static_cast<int>(rv) & index_mask;
        rv = marsaglia_next(rv);
        const int second = static_cast<int>(rv) & index_mask;
        private_data[first] = private_data[first] + term;
        private_data[second] = private_data[second] - term;
      }
      for (int delay = 0; delay < ncs_delay; ++delay) rv = marsaglia_next(rv);
      private_node->m_write_seq = private_node->m_write_seq + 1;
    }

    for (int delay = 0; delay < ncs_delay; ++delay) rv = marsaglia_next(rv);

    if (idelay >= 0) {
      for (int delay = 0; delay < idelay; ++delay) rv = marsaglia_next(rv);
    } else {
      latch::n_pause(static_cast<std::uint32_t>(-idelay));
    }

    if (rdelay != 0) {  // uniform random delay
      rv = marsaglia_next(rv);
      latch::n_pause(static_cast<std::uint32_t>(((rdelay + rdelay) * static_cast<int>(rv & 0x3FF)) >> 10));
    }

    // Geometric random delay: Bernoulli trials until first success.
    while (static_cast<int>((rv = marsaglia_next(rv)) & fixed_mask) < imask) latch::cpu_pause();
  }

  m_its_completed[static_cast<std::size_t>(id)] = its;
  pthread_mutex_lock(&m_gate);
  m_its_by_node[node_id] += its;
  m_node_distribution[node_id] += 1;
  m_lock_migrations_read += migrations_read;
  m_lock_migrations_write += migrations_write;
  m_total_reads += its - writes;
  m_total_writes += writes;
  m_total_raw += raw;
  pthread_mutex_unlock(&m_gate);
  m_halted.fetch_add(1, std::memory_order_release);
}

[[nodiscard]] std::int64_t now_nanos() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec * std::int64_t{1'000'000'000} + now.tv_nsec;
}

[[nodiscard]] std::int64_t micros(const timeval& time) noexcept {
  return time.tv_sec * std::int64_t{1'000'000} + time.tv_usec;
}

template <class Lock>
int RwBench<Lock>::run(const char* executable) {
  // Fisher-Yates, ascending, one pass.
  m_shuffle_map.resize(static_cast<std::size_t>(m_thread_count));
  for (int index = 1; index < m_thread_count; ++index) {
    const auto other = static_cast<int>(latch::next_random() % static_cast<std::uint32_t>(index + 1));
    m_shuffle_map[static_cast<std::size_t>(index)] = m_shuffle_map[static_cast<std::size_t>(other)];
    m_shuffle_map[static_cast<std::size_t>(other)] = index;
  }

  latch::ensure_runtime();  // node map for per-node statistics
  // A native lock never goes through pthread_rwlock_*, whatever is preloaded.
  const char* const preload = std::getenv("LD_PRELOAD");
  const char* const lock_name = m_options.m_kind == LockKind::recursive ? "<recursive>"
                                : m_options.m_kind == LockKind::native  ? "<native>"
                                : preload != nullptr                    ? preload
                                                                        : "<null>";
  std::println("LD_PRELOAD={}", lock_name);

  if (m_options.m_bind != 0) {
    m_placement = build_placement();
    if (m_placement.empty()) fail("no CPUs to bind to", 0);
  }

  if (m_options.m_vstress != 0) {
    std::println("Stress mode; concurrency changes every {} msecs. Ignore the performance results!", m_options.m_vstress);
    pthread_t stress{};
    pthread_create(&stress, nullptr, change_concurrency, this);
    pthread_detach(stress);
  }

  // "Write intensity": 0 = no writes, 1 = saturated, >1 = oversaturated. Bounds the
  // maximum speedup. Does not account for inter-operation delays.
  double write_intensity = 0.0;
  for (const ThreadBlock& block : m_threads) {
    const double effective_write = block.m_wcs_length * (block.m_write_pct / 100.0);
    const double effective_read = block.m_rcs_length * ((100.0 - block.m_write_pct) / 100.0);
    write_intensity += effective_write / (effective_write + effective_read + block.m_ncs_length);
  }
  std::println("Write Intensity={:.3f}", write_intensity);
  std::println("Threads={} ALength={} Instances={}", m_thread_count, m_options.m_alength, m_options.m_instances);

  m_its_completed.assign(static_cast<std::size_t>(m_thread_count), 0);
  const std::size_t table_bytes = static_cast<std::size_t>(m_options.m_instances) * sizeof(RwNode<Lock>*);
  m_table = static_cast<RwNode<Lock>**>(
      std::aligned_alloc(coherence_unit, (table_bytes + coherence_unit - 1) & ~(coherence_unit - 1)));
  for (int index = 0; index < m_options.m_instances; ++index) {
    m_table[index] = RwNode<Lock>::create(m_options.m_alength);
  }

  poll(nullptr, 0, 100);
  for (int id = 0; id < m_thread_count; ++id) launch(this, id);
  while (m_ready.load(std::memory_order_acquire) != m_thread_count) poll(nullptr, 0, 1);

  // Quiesce: lets the scheduler disperse threads and reach steady state, lets them
  // gain affinity, and gives the power governor time to respond.
  poll(nullptr, 0, 1000);
  const std::int64_t start_time = now_nanos();
  rusage before{};
  getrusage(RUSAGE_SELF, &before);
  m_start.store(1, std::memory_order_release);

  poll(nullptr, 0, m_options.m_duration);

  m_halt.store(1, std::memory_order_relaxed);
  const std::int64_t end_time = now_nanos();
  while (m_halted.load(std::memory_order_acquire) != m_thread_count) poll(nullptr, 0, 1);

  std::print("Node distribution: ");
  for (int node = 0; node < latch::runtime.m_node_count; ++node) {
    std::print("{}:{} ", m_node_distribution[node], m_its_by_node[node]);
  }
  std::println("");

  // Post-mortem integrity: writers conserve the sum of each array.
  for (int table_index = 0; table_index < m_options.m_instances; ++table_index) {
    RwNode<Lock>* const node = m_table[table_index];
    int sum = 0;
    for (int index = 0; index < node->m_length; ++index) sum += node->data()[index];
    if (sum != 0) fail("content integrity failure", sum);
  }
  std::println("Post validation : OK");
  std::println("Lock Migration: W={} R={} : Writes={} Reads={} RAW={} : WriteI={:.3f}", m_lock_migrations_write,
               m_lock_migrations_read, m_total_writes, m_total_reads, m_total_raw, write_intensity);

  std::int64_t maximum = 0;
  std::int64_t minimum = -1;
  std::int64_t sum_completed = 0;
  double sum_squares = 0.0;
  for (const std::int64_t value : m_its_completed) {
    sum_completed += value;
    sum_squares += static_cast<double>(value) * static_cast<double>(value);
    maximum = std::max(maximum, value);
    if (value < minimum || minimum < 0) minimum = value;
  }
  const double average = static_cast<double>(sum_completed) / m_thread_count;
  double deviation = 0.0;
  for (const std::int64_t value : m_its_completed) {
    deviation += (static_cast<double>(value) - average) * (static_cast<double>(value) - average);
  }
  deviation = std::sqrt(deviation / m_thread_count);
  std::ranges::sort(m_its_completed);
  // Fixed-time methodology, so watch fairness: Jain's index over raw scores.
  const double jain = (static_cast<double>(sum_completed) * static_cast<double>(sum_completed)) / (m_thread_count * sum_squares);
  const double spread = static_cast<double>(maximum) / static_cast<double>(minimum + 1);
  const int interval_ms = static_cast<int>((end_time - start_time) / 1'000'000);
  std::println("Duration={} Total={}", interval_ms, sum_completed);
  std::println("STATS: Min={} Max={} Sum={} Avg={} Med={} Spread={:.3f} stddev={:.3f} JFI={:.2f}", minimum, maximum,
               sum_completed, sum_completed / m_thread_count, m_its_completed[m_its_completed.size() / 2], spread,
               deviation, jain);

  rusage after{};
  getrusage(RUSAGE_SELF, &after);
  const std::int64_t user_ms = (micros(after.ru_utime) - micros(before.ru_utime)) / 1000;
  const std::int64_t system_ms = (micros(after.ru_stime) - micros(before.ru_stime)) / 1000;
  std::println("User={} System={} msecs in {} msecs", user_ms, system_ms, interval_ms);
  const int parallelism = std::min(m_thread_count, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
  const double utilization = static_cast<double>(user_ms) / (parallelism * interval_ms);

  utsname machine{};
  uname(&machine);
  std::println("SYNOPSIS: {} {} {} {} : {} Ops {:.2f}M Spread={:.2f} Utilization={:.2f}", executable, m_options.m_comment,
               machine.nodename, lock_name, sum_completed, static_cast<double>(sum_completed) / 1e6, spread, utilization);
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  setbuf(stdout, nullptr);
  Options options;
  ThreadBlock ambient;
  std::vector<ThreadBlock> threads;

  struct Binding {
    std::string_view m_key;
    int* m_target;
  };
  const Binding bindings[] = {
      {"Verbose", &options.m_verbose},     {"Instances", &options.m_instances},
      {"ALength", &options.m_alength},     {"_Duration", &options.m_duration},
      {"RDelay", &options.m_rdelay},       {"VStress", &options.m_vstress},
      {"Barrier", &options.m_barrier},     {"Bind", &options.m_bind},
      {"WCSLength", &ambient.m_wcs_length}, {"RCSLength", &ambient.m_rcs_length},
      {"NCSLength", &ambient.m_ncs_length}, {"NCSDelay", &ambient.m_ncs_delay},
      {"RCSDelay", &ambient.m_rcs_delay},  {"WCSDelay", &ambient.m_wcs_delay},
      {"IDelay", &ambient.m_idelay},       {"IMask", &ambient.m_imask},
  };

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument.starts_with('+')) {
      // +<Threads>@<WritePct>: threads pick up the values set so far.
      const auto at = argument.find('@');
      if (at == std::string_view::npos) fail("expected +<Threads>@<WritePct>");
      const int count = std::atoi(argument.substr(1, at - 1).data());
      ThreadBlock block = ambient;
      block.m_write_pct = std::atoi(argument.substr(at + 1).data());
      threads.insert(threads.end(), static_cast<std::size_t>(count), block);
      continue;
    }
    const auto equals = argument.find('=');
    if (equals == std::string_view::npos) {
      std::println("Unknown argument : {}", argument);
      return 1;
    }
    const std::string_view key = argument.substr(0, equals);
    const char* const value = argv[index] + equals + 1;
    if (key == "Comment") {
      options.m_comment = value;
    } else if (key == "Lock") {
      const std::string_view kind = value;
      if (kind == "pthread") {
        options.m_kind = LockKind::pthread;
      } else if (kind == "native") {
        options.m_kind = LockKind::native;
      } else if (kind == "recursive") {
        options.m_kind = LockKind::recursive;
      } else if (kind == "barging") {
        options.m_kind = LockKind::barging;
      } else {
        std::println("Unknown lock : {}", kind);
        return 1;
      }
    } else {
      const auto* const binding = std::ranges::find(bindings, key, &Binding::m_key);
      if (binding == std::end(bindings)) {
        std::println("Unknown key : {}", key);
        return 1;
      }
      *binding->m_target = static_cast<int>(std::strtol(value, nullptr, 0));
    }
  }
  if (threads.empty()) {
    std::println("No threads specified");
    return 1;
  }

  if (options.m_kind == LockKind::recursive) {
    static RwBench<latch::RecursiveRwLock> bench(options, threads);
    return bench.run(argv[0]);
  }
  if (options.m_kind == LockKind::barging) {
    static RwBench<latch::BargingRwLock> bench(options, threads);
    return bench.run(argv[0]);
  }
  if (options.m_kind == LockKind::native) {
    static RwBench<latch::RwLock> bench(options, threads);
    return bench.run(argv[0]);
  }
  static RwBench<PthreadLock> bench(options, threads);
  return bench.run(argv[0]);
}
