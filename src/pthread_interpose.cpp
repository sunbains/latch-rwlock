// SPDX-License-Identifier: MIT OR Apache-2.0
// LD_PRELOAD interposition: runs unmodified binaries on this lock.
//
// Instances are far larger than pthread_rwlock_t, so the first 8 bytes of the
// caller's lock (zero under every initializer) hold a pointer to the real one,
// installed on first use. Attributes are honored in the attribute object we own;
// process-shared locks are refused at init rather than silently broken.
#include <errno.h>
#include <pthread.h>
#include <time.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>

#include "latch/rw_lock.hpp"

using latch::RwLockCore;
using latch::ThreadState;

namespace {

static_assert(sizeof(std::uintptr_t) <= sizeof(pthread_rwlock_t));
static_assert(alignof(pthread_rwlock_t) >= alignof(std::uintptr_t));

// Our view of pthread_rwlockattr_t; every function touching it is interposed.
struct AttrOverlay {
  int m_kind;
  int m_shared;
};
static_assert(sizeof(AttrOverlay) <= sizeof(pthread_rwlockattr_t));

[[gnu::always_inline]] inline std::atomic_ref<std::uintptr_t> inflation_word(pthread_rwlock_t* lock) noexcept {
  return std::atomic_ref<std::uintptr_t>(*reinterpret_cast<std::uintptr_t*>(lock));
}

[[gnu::noinline]] RwLockCore* inflate(pthread_rwlock_t* lock) noexcept {
  RwLockCore* const fresh = RwLockCore::create();
  std::uintptr_t expected = 0;
  if (inflation_word(lock).compare_exchange_strong(expected, reinterpret_cast<std::uintptr_t>(fresh),
                                                   std::memory_order_acq_rel)) {
    return fresh;
  }
  RwLockCore::dispose(fresh);
  return reinterpret_cast<RwLockCore*>(expected);
}

[[gnu::always_inline]] inline RwLockCore* core_of(pthread_rwlock_t* lock) noexcept {
  const std::uintptr_t word = inflation_word(lock).load(std::memory_order_acquire);
  if (word != 0) [[likely]] return reinterpret_cast<RwLockCore*>(word);
  return inflate(lock);
}

[[nodiscard]] timespec now_on(clockid_t clock) noexcept {
  timespec now{};
  clock_gettime(clock, &now);
  return now;
}

// Compared field by field: a deadline far in the future does not fit in nanoseconds.
[[nodiscard]] bool has_passed(const timespec& deadline, const timespec& now) noexcept {
  return now.tv_sec != deadline.tv_sec ? now.tv_sec > deadline.tv_sec : now.tv_nsec >= deadline.tv_nsec;
}

// Ticket queues cannot be abandoned, so timed acquisition polls the try path. Arguments
// are checked first, even if the lock is free, as glibc does.
int poll_deadline(auto attempt, clockid_t clock, const timespec* deadline) noexcept {
  if (clock != CLOCK_REALTIME && clock != CLOCK_MONOTONIC) return EINVAL;
  if (deadline == nullptr || deadline->tv_nsec < 0 || deadline->tv_nsec >= 1'000'000'000) return EINVAL;
  int status = attempt();
  if (status != EBUSY) return status;
  for (long nap = 1'000;; nap = std::min(nap * 2, 1'000'000L)) {
    if (has_passed(*deadline, now_on(clock))) return ETIMEDOUT;
    latch::short_sleep(nap);
    status = attempt();
    if (status != EBUSY) return status;
  }
}

int timed_read(pthread_rwlock_t* lock, clockid_t clock, const timespec* deadline) noexcept {
  ThreadState& self = latch::current_thread();
  RwLockCore* const core = core_of(lock);
  if (core->is_write_holder(self)) return EDEADLK;
  return poll_deadline([&] { return core->try_read_acquire(self); }, clock, deadline);
}

int timed_write(pthread_rwlock_t* lock, clockid_t clock, const timespec* deadline) noexcept {
  ThreadState& self = latch::current_thread();
  RwLockCore* const core = core_of(lock);
  if (core->held_by(self)) return EDEADLK;
  return poll_deadline([&] { return core->try_write_acquire(self); }, clock, deadline);
}

// An invalid tv_nsec passes through, for poll_deadline to refuse.
[[nodiscard]] timespec relative_deadline(const timespec& relative) noexcept {
  if (relative.tv_nsec < 0 || relative.tv_nsec >= 1'000'000'000) return relative;
  const timespec now = now_on(CLOCK_MONOTONIC);
  timespec deadline{};
  deadline.tv_nsec = now.tv_nsec + relative.tv_nsec;
  const bool carry = deadline.tv_nsec >= 1'000'000'000;
  if (carry) deadline.tv_nsec -= 1'000'000'000;
  if (__builtin_add_overflow(now.tv_sec, relative.tv_sec, &deadline.tv_sec) ||
      __builtin_add_overflow(deadline.tv_sec, time_t{carry}, &deadline.tv_sec)) {
    deadline.tv_sec = std::numeric_limits<time_t>::max();
  }
  return deadline;
}

}  // namespace

#define LATCH_EXPORT [[gnu::visibility("default")]]

extern "C" {

LATCH_EXPORT int pthread_rwlock_init(pthread_rwlock_t* __restrict lock,
                                      const pthread_rwlockattr_t* __restrict attr) noexcept {
  if (attr != nullptr && reinterpret_cast<const AttrOverlay*>(attr)->m_shared == PTHREAD_PROCESS_SHARED) {
    return ENOTSUP;
  }
  std::memset(static_cast<void*>(lock), 0, sizeof(*lock));
  return 0;
}

LATCH_EXPORT int pthread_rwlock_destroy(pthread_rwlock_t* lock) noexcept {
  const std::uintptr_t word = inflation_word(lock).load(std::memory_order_acquire);
  if (word == 0) return 0;
  auto* const core = reinterpret_cast<RwLockCore*>(word);
  if (!core->is_quiescent()) {
    if (latch::runtime.m_warn) latch::diagnostic("Error: destroying busy rwlock %p\n", static_cast<void*>(lock));
    return EBUSY;
  }
  inflation_word(lock).store(0, std::memory_order_release);
  RwLockCore::dispose(core);
  return 0;
}

LATCH_EXPORT int pthread_rwlock_rdlock(pthread_rwlock_t* lock) noexcept {
  ThreadState& self = latch::current_thread();
  return core_of(lock)->read_acquire(self);
}

LATCH_EXPORT int pthread_rwlock_wrlock(pthread_rwlock_t* lock) noexcept {
  ThreadState& self = latch::current_thread();
  return core_of(lock)->write_acquire(self);
}

LATCH_EXPORT int pthread_rwlock_unlock(pthread_rwlock_t* lock) noexcept {
  ThreadState& self = latch::current_thread();
  const std::uintptr_t word = inflation_word(lock).load(std::memory_order_acquire);
  if (word == 0) [[unlikely]] {
    if (latch::runtime.m_warn) latch::diagnostic("Warning: unlocking idle rwlock %p\n", static_cast<void*>(lock));
    return EPERM;
  }
  return reinterpret_cast<RwLockCore*>(word)->release(self);
}

LATCH_EXPORT int pthread_rwlock_tryrdlock(pthread_rwlock_t* lock) noexcept {
  ThreadState& self = latch::current_thread();
  return core_of(lock)->try_read_acquire(self);
}

LATCH_EXPORT int pthread_rwlock_trywrlock(pthread_rwlock_t* lock) noexcept {
  ThreadState& self = latch::current_thread();
  return core_of(lock)->try_write_acquire(self);
}

LATCH_EXPORT int pthread_rwlock_timedrdlock(pthread_rwlock_t* __restrict lock,
                                             const timespec* __restrict deadline) noexcept {
  return timed_read(lock, CLOCK_REALTIME, deadline);
}

LATCH_EXPORT int pthread_rwlock_timedwrlock(pthread_rwlock_t* __restrict lock,
                                             const timespec* __restrict deadline) noexcept {
  return timed_write(lock, CLOCK_REALTIME, deadline);
}

LATCH_EXPORT int pthread_rwlock_clockrdlock(pthread_rwlock_t* __restrict lock, clockid_t clock,
                                             const timespec* __restrict deadline) noexcept {
  return timed_read(lock, clock, deadline);
}

LATCH_EXPORT int pthread_rwlock_clockwrlock(pthread_rwlock_t* __restrict lock, clockid_t clock,
                                             const timespec* __restrict deadline) noexcept {
  return timed_write(lock, clock, deadline);
}

LATCH_EXPORT int pthread_rwlock_reltimedrdlock_np(pthread_rwlock_t* __restrict lock,
                                                   const timespec* __restrict relative) noexcept {
  const timespec deadline = relative_deadline(*relative);
  return timed_read(lock, CLOCK_MONOTONIC, &deadline);
}

LATCH_EXPORT int pthread_rwlock_reltimedwrlock_np(pthread_rwlock_t* __restrict lock,
                                                   const timespec* __restrict relative) noexcept {
  const timespec deadline = relative_deadline(*relative);
  return timed_write(lock, CLOCK_MONOTONIC, &deadline);
}

LATCH_EXPORT int pthread_rwlockattr_init(pthread_rwlockattr_t* attr) noexcept {
  std::memset(static_cast<void*>(attr), 0, sizeof(*attr));
  reinterpret_cast<AttrOverlay*>(attr)->m_kind = PTHREAD_RWLOCK_DEFAULT_NP;
  reinterpret_cast<AttrOverlay*>(attr)->m_shared = PTHREAD_PROCESS_PRIVATE;
  return 0;
}

LATCH_EXPORT int pthread_rwlockattr_destroy(pthread_rwlockattr_t*) noexcept { return 0; }

LATCH_EXPORT int pthread_rwlockattr_getpshared(const pthread_rwlockattr_t* __restrict attr,
                                                int* __restrict shared) noexcept {
  *shared = reinterpret_cast<const AttrOverlay*>(attr)->m_shared;
  return 0;
}

LATCH_EXPORT int pthread_rwlockattr_setpshared(pthread_rwlockattr_t* attr, int shared) noexcept {
  if (shared != PTHREAD_PROCESS_PRIVATE && shared != PTHREAD_PROCESS_SHARED) return EINVAL;
  reinterpret_cast<AttrOverlay*>(attr)->m_shared = shared;
  return 0;
}

// Kinds are accepted for compatibility. This lock always prefers writers, and
// recursive read acquisition is safe under any kind because it is tracked per thread.
LATCH_EXPORT int pthread_rwlockattr_getkind_np(const pthread_rwlockattr_t* __restrict attr,
                                                int* __restrict kind) noexcept {
  *kind = reinterpret_cast<const AttrOverlay*>(attr)->m_kind;
  return 0;
}

LATCH_EXPORT int pthread_rwlockattr_setkind_np(pthread_rwlockattr_t* attr, int kind) noexcept {
  if (kind < PTHREAD_RWLOCK_PREFER_READER_NP || kind > PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP) return EINVAL;
  reinterpret_cast<AttrOverlay*>(attr)->m_kind = kind;
  return 0;
}

// glibc's internal aliases, reachable from older binaries.
LATCH_EXPORT [[gnu::alias("pthread_rwlock_init")]] int __pthread_rwlock_init(pthread_rwlock_t*, const pthread_rwlockattr_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_destroy")]] int __pthread_rwlock_destroy(pthread_rwlock_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_rdlock")]] int __pthread_rwlock_rdlock(pthread_rwlock_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_wrlock")]] int __pthread_rwlock_wrlock(pthread_rwlock_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_unlock")]] int __pthread_rwlock_unlock(pthread_rwlock_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_tryrdlock")]] int __pthread_rwlock_tryrdlock(pthread_rwlock_t*) noexcept;
LATCH_EXPORT [[gnu::alias("pthread_rwlock_trywrlock")]] int __pthread_rwlock_trywrlock(pthread_rwlock_t*) noexcept;

}  // extern "C"

// Initialize early, while almost certainly single-threaded, and register the
// primordial thread. Every path also initializes on demand, in case another
// library's constructor reaches a rwlock before this runs.
[[gnu::constructor]] static void latch_library_init() {
  latch::ensure_runtime();
  (void)latch::current_thread();
}
