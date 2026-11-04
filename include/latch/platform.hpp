// SPDX-License-Identifier: MIT OR Apache-2.0
// Platform layer: what std::atomic cannot express.
#pragma once

#include <sched.h>
#include <time.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "latch/config.hpp"

#if __has_include(<sys/rseq.h>)
#include <sys/rseq.h>
#define LATCH_HAVE_RSEQ 1
#else
#define LATCH_HAVE_RSEQ 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#if defined(__SANITIZE_THREAD__)
#define LATCH_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LATCH_TSAN 1
#endif
#endif
#ifndef LATCH_TSAN
#define LATCH_TSAN 0
#endif

namespace latch {

// Effective coherence unit. x86 lines are 64 bytes, but the adjacent-sector
// prefetcher makes pairs of lines behave as one unit for false sharing.
#if defined(__x86_64__) || defined(__i386__) || defined(__aarch64__)
inline constexpr std::size_t coherence_unit = 128;
#else
inline constexpr std::size_t coherence_unit = 64;
#endif

// Lock instances hold arrays indexed by NUMA node. Machines with more nodes
// than this fold several nodes onto one index rather than refusing to start.
inline constexpr int max_numa_nodes = 8;

// Deliberately no "memory" clobber: PAUSE is not a sequence point. Waiting loops
// re-read through std::atomic, which handles compile-time reordering.
[[gnu::always_inline]] inline void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("isb");
#endif
}

// Partially unrolled to reduce loop-branch overhead per pause.
inline void n_pause(std::uint32_t count) noexcept {
  while (count >= 4) {
    cpu_pause();
    cpu_pause();
    cpu_pause();
    cpu_pause();
    count -= 4;
  }
  while (count-- != 0) cpu_pause();
}

// StoreLoad barrier. On x86 a locked add to the stack is cheaper than MFENCE;
// below the stack pointer so it cannot contend with the frame's own accesses.
[[gnu::always_inline]] inline void store_load_fence() noexcept {
#if (defined(__x86_64__)) && !LATCH_TSAN
  asm volatile("lock; addl $0,-8(%%rsp)" ::: "memory", "cc");
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// Prefetch for ownership. Going straight to exclusive avoids a shared-then-upgrade
// pair of coherence transactions ahead of a CAS. CPUs without PREFETCHW execute
// it as a NOP.
[[gnu::always_inline]] inline void prefetch_for_write(const volatile void* address) noexcept {
#if defined(__x86_64__)
  asm volatile("prefetchw %0" :: "m"(*static_cast<const volatile char*>(address)));
#else
  __builtin_prefetch(const_cast<const void*>(address), 1);
#endif
}

[[gnu::always_inline]] inline std::uint64_t read_clock() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  return __rdtsc();
#elif defined(__aarch64__)
  std::uint64_t value;
  asm volatile("mrs %0, cntvct_el0" : "=r"(value));
  return value;
#else
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
#endif
}

// Identify the CPU we are running on. glibc >= 2.35 registers rseq for each
// thread, so the kernel keeps the current CPU in a thread-local word: one load,
// no VDSO call. RDTSCP:ECX remains selectable with LATCH_CPU_SOURCE=2.
[[gnu::always_inline]] inline int current_cpu() noexcept {
#if LATCH_CPU_SOURCE == 2 && defined(__x86_64__)
  unsigned int aux = 0;
  (void)__rdtscp(&aux);
  return static_cast<int>(aux & 0xFFF);
#elif LATCH_CPU_SOURCE == 0 && LATCH_HAVE_RSEQ
  const auto* area = reinterpret_cast<const volatile struct rseq*>(
      static_cast<const char*>(__builtin_thread_pointer()) + __rseq_offset);
  const auto cpu = static_cast<std::int32_t>(area->cpu_id);
  if (cpu >= 0) [[likely]] return cpu;
  return sched_getcpu();  // rseq not registered for this thread
#else
  return sched_getcpu();
#endif
}

}  // namespace latch
