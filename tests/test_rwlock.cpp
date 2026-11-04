// SPDX-License-Identifier: MIT OR Apache-2.0
// Behavior and stress tests. Run as:
//   test_rwlock native                 RwLock through its C++ interface
//   LD_PRELOAD=... test_rwlock pthread the interposed POSIX API
//
// Oversubscribed cases run with far more threads than CPUs on purpose.
#include <errno.h>
#include <pthread.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <print>
#include <shared_mutex>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "latch/latch.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::println("FAILED {}:{}: {}", __FILE__, __LINE__, #condition);      \
      ++failures;                                                            \
    }                                                                        \
  } while (false)

using namespace std::chrono_literals;

const unsigned cpu_count = std::thread::hardware_concurrency();

// std::shared_lock etc. compile against the native type.
static_assert(requires(latch::RwLock& lock) {
  lock.lock();
  lock.unlock();
  lock.lock_shared();
  lock.unlock_shared();
  { lock.try_lock() } -> std::same_as<bool>;
  { lock.try_lock_shared() } -> std::same_as<bool>;
  { lock.try_lock_for(1ms) } -> std::same_as<bool>;
  { lock.try_lock_shared_until(std::chrono::steady_clock::now()) } -> std::same_as<bool>;
});

// Exclusion checked with a seqlock-style counter plus a conservation invariant.
template <class Lock>
void exclusion_stress(Lock& lock, unsigned thread_count, int write_pct, std::chrono::milliseconds duration) {
  long data[16] = {};
  volatile int sequence = 0;
  std::atomic<bool> stop{false};
  std::atomic<long> operations{0};
  std::atomic<int> violations{0};
  std::vector<std::thread> threads;
  for (unsigned id = 0; id < thread_count; ++id) {
    threads.emplace_back([&, id] {
      std::uint32_t random = id + 1;
      long count = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        random = latch::marsaglia_next(random);
        if (static_cast<int>(random % 100) < write_pct) {
          lock.lock();
          if ((sequence & 1) != 0) violations.fetch_add(1);
          sequence = sequence + 1;
          data[random & 15] += 7;
          data[(random >> 8) & 15] -= 7;
          sequence = sequence + 1;
          lock.unlock();
        } else {
          lock.lock_shared();
          const int before = sequence;
          long sum = 0;
          for (const long value : data) sum += value;
          if (before != sequence || (before & 1) != 0 || sum != 0) violations.fetch_add(1);
          lock.unlock_shared();
        }
        ++count;
      }
      operations.fetch_add(count);
    });
  }
  std::this_thread::sleep_for(duration);
  stop.store(true);
  for (auto& thread : threads) thread.join();
  CHECK(violations.load() == 0);
  CHECK(operations.load() > 0);
  std::println("  stress threads={} write%={}: {} ops, {} violations", thread_count, write_pct, operations.load(), violations.load());
}

// Every thread must get in repeatedly, however many there are relative to CPUs.
template <class Lock>
void progress_check(Lock& lock, unsigned thread_count) {
  std::atomic<unsigned> finished{0};
  std::vector<std::thread> threads;
  for (unsigned id = 0; id < thread_count; ++id) {
    threads.emplace_back([&, id] {
      for (int round = 0; round < 50; ++round) {
        if ((id + round) % 4 == 0) {
          lock.lock();
          lock.unlock();
        } else {
          lock.lock_shared();
          lock.unlock_shared();
        }
      }
      finished.fetch_add(1);
    });
  }
  for (auto& thread : threads) thread.join();
  CHECK(finished.load() == thread_count);
}

bool refused_as_deadlock(auto attempt) {
  try {
    attempt();
  } catch (const std::system_error& error) {
    return error.code() == std::errc::resource_deadlock_would_occur;
  }
  return false;
}

void test_self_deadlock(latch::RwLock& lock) {
  const auto others_can = [&](auto attempt) {
    bool result = false;
    std::thread([&] { result = attempt(); }).join();
    return result;
  };
  const auto try_read = [&] {
    const bool taken = lock.try_lock_shared();
    if (taken) lock.unlock_shared();
    return taken;
  };
  const auto try_write = [&] {
    const bool taken = lock.try_lock();
    if (taken) lock.unlock();
    return taken;
  };

  // Direct calls.
  lock.lock();
  CHECK(refused_as_deadlock([&] { lock.lock(); }));
  CHECK(refused_as_deadlock([&] { lock.lock_shared(); }));
  CHECK(!others_can(try_read));  // the refusals left the write lock in place
  lock.unlock();
  CHECK(others_can(try_write));

  lock.lock_shared();
  CHECK(refused_as_deadlock([&] { lock.lock(); }));
  CHECK(others_can(try_read));
  CHECK(!others_can(try_write));  // the refusal left the read lock in place
  lock.unlock_shared();
  CHECK(others_can(try_write));

  // Guards: a refused inner guard must not release what the outer one holds.
  {
    std::unique_lock outer(lock);
    CHECK(refused_as_deadlock([&] { std::unique_lock inner(lock); }));
    CHECK(refused_as_deadlock([&] { std::shared_lock inner(lock); }));
    CHECK(refused_as_deadlock([&] { std::lock_guard inner(lock); }));
    CHECK(outer.owns_lock());
    CHECK(!others_can(try_read));
  }
  CHECK(others_can(try_write));
  {
    std::shared_lock outer(lock);
    CHECK(refused_as_deadlock([&] { std::unique_lock inner(lock); }));
    CHECK(outer.owns_lock());
    CHECK(others_can(try_read));
    CHECK(!others_can(try_write));
    {
      std::shared_lock nested(lock);  // allowed
      CHECK(nested.owns_lock());
    }
    CHECK(!others_can(try_write));
  }
  CHECK(others_can(try_write));
}

void test_native() {
  std::println("native: self-deadlock throws, directly and through guards");
  {
    latch::RwLock adaptive;
    test_self_deadlock(adaptive);
    latch::RwLock no_fast_reads{latch::ExpectedWrites{100}};
    test_self_deadlock(no_fast_reads);
  }

  std::println("native: a failed try_lock must not lose sight of a fast-path reader");
  {
    latch::RwLock lock;
    lock.lock_shared();  // the first, slow reader enables the bias
    lock.unlock_shared();
    std::atomic<bool> holding{false};
    std::atomic<bool> finish{false};
    std::thread reader([&] {
      lock.lock_shared();
      holding.store(true);
      while (!finish.load()) std::this_thread::sleep_for(1ms);
      lock.unlock_shared();
    });
    while (!holding.load()) std::this_thread::yield();
    CHECK(!lock.try_lock());
    CHECK(!lock.try_lock());
    CHECK(!lock.try_lock_for(5ms));
    std::atomic<bool> entered{false};
    std::thread writer([&] {
      lock.lock();
      entered.store(true);
      lock.unlock();
    });
    std::this_thread::sleep_for(50ms);
    CHECK(!entered.load());
    finish.store(true);
    reader.join();
    writer.join();
    CHECK(entered.load());
  }

  std::println("native: try_lock writers exclude fast-path readers");
  {
    latch::RwLock lock;
    std::atomic<int> readers_inside{0};
    std::atomic<int> writers_inside{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < 6; ++index) {
      threads.emplace_back([&, index] {
        while (!stop.load(std::memory_order_relaxed)) {
          if (index % 3 == 0) {
            if (lock.try_lock()) {
              CHECK(writers_inside.fetch_add(1) == 0);
              CHECK(readers_inside.load() == 0);
              writers_inside.fetch_sub(1);
              lock.unlock();
            }
          } else {
            lock.lock_shared();
            readers_inside.fetch_add(1);
            CHECK(writers_inside.load() == 0);
            readers_inside.fetch_sub(1);
            lock.unlock_shared();
          }
        }
      });
    }
    std::this_thread::sleep_for(500ms);
    stop.store(true);
    for (auto& thread : threads) thread.join();
  }

  std::println("native: guards");
  {
    latch::RwLock lock;
    {
      std::shared_lock first(lock);
      std::shared_lock second(lock);  // nesting
      CHECK(first.owns_lock() && second.owns_lock());
    }
    {
      std::unique_lock exclusive(lock);
      std::thread([&] {
        CHECK(!lock.try_lock());
        CHECK(!lock.try_lock_shared());
        CHECK(!lock.try_lock_for(5ms));
        CHECK(!lock.try_lock_shared_for(5ms));
      }).join();
    }
    CHECK(lock.try_lock());
    lock.unlock();
    CHECK(lock.try_lock_shared());
    std::thread([&] {
      CHECK(!lock.try_lock());
      CHECK(lock.try_lock_shared());
      lock.unlock_shared();
    }).join();
    lock.unlock_shared();
  }

  std::println("native: nested read with a pending writer must not deadlock");
  for (int round = 0; round < 200; ++round) {
    latch::RwLock lock;
    lock.lock_shared();
    std::atomic<bool> started{false};
    std::thread writer([&] {
      started.store(true);
      lock.lock();
      lock.unlock();
    });
    while (!started.load()) std::this_thread::yield();
    if (round % 2 == 0) std::this_thread::sleep_for(100us);
    lock.lock_shared();
    lock.unlock_shared();
    lock.unlock_shared();
    writer.join();
  }

  std::println("native: a thread holding many read locks");
  {
    std::vector<std::unique_ptr<latch::RwLock>> locks;
    for (int index = 0; index < 64; ++index) locks.push_back(std::make_unique<latch::RwLock>());
    for (int round = 0; round < 3; ++round) {
      for (auto& lock : locks) lock->lock_shared();
      std::thread([&] {
        for (auto& lock : locks) CHECK(!lock->try_lock());
      }).join();
      for (auto& lock : locks) lock->unlock_shared();
      for (auto& lock : locks) {
        lock->lock();
        lock->unlock();
      }
    }
  }

  std::println("native: bias revocation with a held fast-path read");
  {
    latch::RwLock lock;
    for (int index = 0; index < 4; ++index) {  // first reads enable the bias
      lock.lock_shared();
      lock.unlock_shared();
    }
    lock.lock_shared();
    std::atomic<bool> acquired{false};
    std::thread writer([&] {
      lock.lock();
      acquired.store(true);
      lock.unlock();
    });
    std::this_thread::sleep_for(20ms);
    CHECK(!acquired.load());
    lock.unlock_shared();
    writer.join();
    CHECK(acquired.load());
  }

  std::println("native: thread churn recycles reader rows");
  {
    latch::RwLock lock;
    for (int round = 0; round < 2000; ++round) {
      std::thread([&] {
        lock.lock_shared();
        lock.unlock_shared();
      }).join();
    }
    lock.lock();
    lock.unlock();
  }

  std::println("native: writer sleeping behind a long reader is woken");
  {
    latch::RwLock lock;
    lock.lock_shared();
    std::thread writer([&] {
      lock.lock();
      lock.unlock();
    });
    std::this_thread::sleep_for(200ms);  // long past patience
    lock.unlock_shared();
    writer.join();
  }

  std::println("native: readers sleeping behind a long writer are woken");
  {
    latch::RwLock lock;
    lock.lock();
    std::vector<std::thread> readers;
    for (int index = 0; index < 32; ++index) {
      readers.emplace_back([&] {
        lock.lock_shared();
        lock.unlock_shared();
      });
    }
    std::this_thread::sleep_for(200ms);
    lock.unlock();
    for (auto& reader : readers) reader.join();
  }

  std::println("native: writers sleeping in the queue are woken");
  {
    latch::RwLock lock;
    lock.lock();
    std::vector<std::thread> writers;
    for (int index = 0; index < 32; ++index) {
      writers.emplace_back([&] {
        lock.lock();
        lock.unlock();
      });
    }
    std::this_thread::sleep_for(200ms);
    lock.unlock();
    for (auto& writer : writers) writer.join();
  }

  std::println("native: fixed configuration");
  {
    using latch::ExpectedWrites;
    using latch::LoadProfile;
    latch::RwLock adaptive;
    CHECK(adaptive.core().is_adaptive());
    CHECK(adaptive.core().load_profile() == LoadProfile::read_mostly);
    latch::RwLock light{ExpectedWrites{1}};
    CHECK(!light.core().is_adaptive());
    CHECK(light.core().load_profile() == LoadProfile::read_mostly);
    latch::RwLock mixed{ExpectedWrites{10}};
    CHECK(mixed.core().load_profile() == LoadProfile::mixed);
    latch::RwLock heavy{ExpectedWrites{50}};
    CHECK(!heavy.core().is_adaptive());
    CHECK(heavy.core().load_profile() == LoadProfile::write_heavy);
    // A fixed lock never moves, whatever the load.
    for (int round = 0; round < 200'000; ++round) {
      light.lock();
      light.unlock();
    }
    CHECK(light.core().load_profile() == LoadProfile::read_mostly);
    exclusion_stress(light, cpu_count / 2, 50, 200ms);
    exclusion_stress(heavy, cpu_count / 2, 1, 200ms);
    CHECK(heavy.core().load_profile() == LoadProfile::write_heavy);
  }

  std::println("native: automatic adjustment follows the load in both directions");
  {
    using latch::LoadProfile;
    latch::RwLock lock;
    const auto run_mix = [&lock](int write_pct, std::chrono::milliseconds duration) {
      const auto deadline = std::chrono::steady_clock::now() + duration;
      std::uint32_t random = 7;
      while (std::chrono::steady_clock::now() < deadline) {
        for (int step = 0; step < 512; ++step) {
          random = latch::marsaglia_next(random);
          if (static_cast<int>(random % 100) < write_pct) {
            lock.lock();
            lock.unlock();
          } else {
            lock.lock_shared();
            lock.unlock_shared();
          }
        }
      }
    };
    run_mix(60, 300ms);
    CHECK(lock.core().load_profile() == LoadProfile::write_heavy);
    run_mix(1, 300ms);
    CHECK(lock.core().load_profile() == LoadProfile::read_mostly);
    run_mix(10, 300ms);
    CHECK(lock.core().load_profile() == LoadProfile::mixed);
    run_mix(100, 300ms);
    CHECK(lock.core().load_profile() == LoadProfile::write_heavy);
    run_mix(25, 300ms);  // inside the band: stays where it was
    CHECK(lock.core().load_profile() == LoadProfile::write_heavy);
  }

  std::println("native: more threads than CPUs always selects read-mostly");
  {
    latch::RwLock lock;
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (unsigned id = 0; id < cpu_count * 2; ++id) {
      threads.emplace_back([&] {
        while (!stop.load(std::memory_order_relaxed)) {
          lock.lock();
          lock.unlock();
        }
      });
    }
    std::this_thread::sleep_for(500ms);
    stop.store(true);
    for (auto& thread : threads) thread.join();
    CHECK(lock.core().load_profile() == latch::LoadProfile::read_mostly);
  }

  std::println("native: stress and oversubscription");
  for (const int write_pct : {0, 1, 10, 50, 100}) {
    latch::RwLock lock;
    exclusion_stress(lock, cpu_count / 2, write_pct, 500ms);
    exclusion_stress(lock, cpu_count * 2, write_pct, 500ms);
    exclusion_stress(lock, cpu_count * 4, write_pct, 500ms);
  }
  {
    latch::RwLock lock;
    progress_check(lock, cpu_count * 4);
  }
}

class PosixLock {
 public:
  PosixLock() { pthread_rwlock_init(&m_lock, nullptr); }
  ~PosixLock() { CHECK(pthread_rwlock_destroy(&m_lock) == 0); }
  void lock() { pthread_rwlock_wrlock(&m_lock); }
  void unlock() { pthread_rwlock_unlock(&m_lock); }
  void lock_shared() { pthread_rwlock_rdlock(&m_lock); }
  void unlock_shared() { pthread_rwlock_unlock(&m_lock); }
  pthread_rwlock_t m_lock;
};

[[nodiscard]] timespec deadline_in(clockid_t clock, long milliseconds) {
  timespec now{};
  clock_gettime(clock, &now);
  now.tv_nsec += milliseconds * 1'000'000;
  now.tv_sec += now.tv_nsec / 1'000'000'000;
  now.tv_nsec %= 1'000'000'000;
  return now;
}

void test_pthread() {
  std::println("pthread: static initializer, init, destroy, reinit");
  static pthread_rwlock_t static_lock = PTHREAD_RWLOCK_INITIALIZER;
  CHECK(pthread_rwlock_rdlock(&static_lock) == 0);
  CHECK(pthread_rwlock_unlock(&static_lock) == 0);
  CHECK(pthread_rwlock_wrlock(&static_lock) == 0);
  CHECK(pthread_rwlock_unlock(&static_lock) == 0);

  pthread_rwlock_t lock;
  for (int round = 0; round < 3; ++round) {
    CHECK(pthread_rwlock_init(&lock, nullptr) == 0);
    CHECK(pthread_rwlock_rdlock(&lock) == 0);
    CHECK(pthread_rwlock_unlock(&lock) == 0);
    CHECK(pthread_rwlock_destroy(&lock) == 0);
  }
  CHECK(pthread_rwlock_init(&lock, nullptr) == 0);

  std::println("pthread: self-deadlock detection");
  CHECK(pthread_rwlock_wrlock(&lock) == 0);
  CHECK(pthread_rwlock_wrlock(&lock) == EDEADLK);
  CHECK(pthread_rwlock_rdlock(&lock) == EDEADLK);
  CHECK(pthread_rwlock_trywrlock(&lock) == EBUSY);
  CHECK(pthread_rwlock_tryrdlock(&lock) == EBUSY);
  timespec soon = deadline_in(CLOCK_REALTIME, 5);
  CHECK(pthread_rwlock_timedwrlock(&lock, &soon) == EDEADLK);
  CHECK(pthread_rwlock_unlock(&lock) == 0);
  CHECK(pthread_rwlock_rdlock(&lock) == 0);
  CHECK(pthread_rwlock_wrlock(&lock) == EDEADLK);
  CHECK(pthread_rwlock_unlock(&lock) == 0);

  std::println("pthread: try and timed forms");
  CHECK(pthread_rwlock_wrlock(&lock) == 0);
  std::thread([&] {
    CHECK(pthread_rwlock_tryrdlock(&lock) == EBUSY);
    CHECK(pthread_rwlock_trywrlock(&lock) == EBUSY);
    timespec deadline = deadline_in(CLOCK_REALTIME, 10);
    CHECK(pthread_rwlock_timedrdlock(&lock, &deadline) == ETIMEDOUT);
    deadline = deadline_in(CLOCK_REALTIME, 10);
    CHECK(pthread_rwlock_timedwrlock(&lock, &deadline) == ETIMEDOUT);
    deadline = deadline_in(CLOCK_MONOTONIC, 10);
    CHECK(pthread_rwlock_clockrdlock(&lock, CLOCK_MONOTONIC, &deadline) == ETIMEDOUT);
    deadline = deadline_in(CLOCK_MONOTONIC, 10);
    CHECK(pthread_rwlock_clockwrlock(&lock, CLOCK_MONOTONIC, &deadline) == ETIMEDOUT);
    timespec invalid{0, 2'000'000'000};
    CHECK(pthread_rwlock_timedwrlock(&lock, &invalid) == EINVAL);
  }).join();
  CHECK(pthread_rwlock_unlock(&lock) == 0);
  timespec later = deadline_in(CLOCK_REALTIME, 1000);
  CHECK(pthread_rwlock_timedrdlock(&lock, &later) == 0);
  CHECK(pthread_rwlock_unlock(&lock) == 0);
  later = deadline_in(CLOCK_REALTIME, 1000);
  CHECK(pthread_rwlock_timedwrlock(&lock, &later) == 0);
  CHECK(pthread_rwlock_unlock(&lock) == 0);

  std::println("pthread: try after a writer finished must not fail spuriously");
  for (int round = 0; round < 1000; ++round) {
    CHECK(pthread_rwlock_wrlock(&lock) == 0);
    CHECK(pthread_rwlock_unlock(&lock) == 0);
    CHECK(pthread_rwlock_tryrdlock(&lock) == 0);
    CHECK(pthread_rwlock_unlock(&lock) == 0);
    CHECK(pthread_rwlock_trywrlock(&lock) == 0);
    CHECK(pthread_rwlock_unlock(&lock) == 0);
  }

  std::println("pthread: destroy on a held lock reports EBUSY");
  CHECK(pthread_rwlock_rdlock(&lock) == 0);
  CHECK(pthread_rwlock_destroy(&lock) == EBUSY);
  CHECK(pthread_rwlock_unlock(&lock) == 0);
  CHECK(pthread_rwlock_destroy(&lock) == 0);

  std::println("pthread: attributes");
  pthread_rwlockattr_t attr;
  CHECK(pthread_rwlockattr_init(&attr) == 0);
  int value = -1;
  CHECK(pthread_rwlockattr_getpshared(&attr, &value) == 0 && value == PTHREAD_PROCESS_PRIVATE);
  CHECK(pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP) == 0);
  CHECK(pthread_rwlockattr_getkind_np(&attr, &value) == 0 && value == PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
  CHECK(pthread_rwlock_init(&lock, &attr) == 0);
  CHECK(pthread_rwlock_destroy(&lock) == 0);
  CHECK(pthread_rwlockattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) == 0);
  CHECK(pthread_rwlock_init(&lock, &attr) == ENOTSUP);
  CHECK(pthread_rwlockattr_destroy(&attr) == 0);

  std::println("pthread: libstdc++ std::shared_mutex and std::shared_timed_mutex");
  {
    std::shared_mutex mutex;
    { std::unique_lock exclusive(mutex); }
    { std::shared_lock shared(mutex); }
    std::shared_timed_mutex timed;
    CHECK(timed.try_lock_for(10ms));
    std::thread([&] { CHECK(!timed.try_lock_shared_for(5ms)); }).join();
    timed.unlock();
  }

  std::println("pthread: fork with readers published in the parent");
  {
    PosixLock shared_lock;
    for (int index = 0; index < 4; ++index) {
      shared_lock.lock_shared();
      shared_lock.unlock_shared();
    }
    std::atomic<bool> hold{true};
    std::atomic<bool> holding{false};
    std::thread reader([&] {
      shared_lock.lock_shared();
      holding.store(true);
      while (hold.load()) std::this_thread::sleep_for(1ms);
      shared_lock.unlock_shared();
    });
    while (!holding.load()) std::this_thread::yield();
    PosixLock other;
    other.lock_shared();
    other.unlock_shared();
    const pid_t child = fork();
    if (child == 0) {
      // Bias machinery, membarrier and allocation must all work in the child.
      alarm(30);
      pthread_rwlock_t fresh = PTHREAD_RWLOCK_INITIALIZER;
      for (int index = 0; index < 100; ++index) {
        if (pthread_rwlock_rdlock(&fresh) != 0 || pthread_rwlock_unlock(&fresh) != 0) _exit(2);
        if (pthread_rwlock_wrlock(&fresh) != 0 || pthread_rwlock_unlock(&fresh) != 0) _exit(3);
      }
      if (pthread_rwlock_wrlock(&other.m_lock) != 0) _exit(4);
      _exit(0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    hold.store(false);
    reader.join();
  }

  std::println("pthread: a failed trywrlock must not lose sight of a fast-path reader");
  {
    PosixLock posix;
    CHECK(pthread_rwlock_rdlock(&posix.m_lock) == 0);  // the first, slow reader enables the bias
    CHECK(pthread_rwlock_unlock(&posix.m_lock) == 0);
    std::atomic<bool> holding{false};
    std::atomic<bool> finish{false};
    std::thread reader([&] {
      CHECK(pthread_rwlock_rdlock(&posix.m_lock) == 0);
      holding.store(true);
      while (!finish.load()) std::this_thread::sleep_for(1ms);
      CHECK(pthread_rwlock_unlock(&posix.m_lock) == 0);
    });
    while (!holding.load()) std::this_thread::yield();
    CHECK(pthread_rwlock_trywrlock(&posix.m_lock) == EBUSY);
    CHECK(pthread_rwlock_trywrlock(&posix.m_lock) == EBUSY);
    timespec deadline = deadline_in(CLOCK_REALTIME, 5);
    CHECK(pthread_rwlock_timedwrlock(&posix.m_lock, &deadline) == ETIMEDOUT);
    finish.store(true);
    reader.join();
  }

  std::println("pthread: stress and oversubscription");
  for (const int write_pct : {0, 1, 10, 50, 100}) {
    PosixLock posix;
    exclusion_stress(posix, cpu_count / 2, write_pct, 500ms);
    exclusion_stress(posix, cpu_count * 4, write_pct, 500ms);
  }
  {
    PosixLock posix;
    progress_check(posix, cpu_count * 4);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  setbuf(stdout, nullptr);
  alarm(600);  // a deadlock must not wedge the run
  const std::string_view mode = argc > 1 ? argv[1] : "native";
  if (mode == "pthread") {
    test_pthread();
  } else {
    test_native();
  }
  std::println("{}: {}", mode, failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
