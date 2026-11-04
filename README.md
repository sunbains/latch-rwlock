# latch-rwlock

A header-only, NUMA-aware reader-writer lock for C++26 on Linux, designed to keep
working when threads outnumber CPUs. It is built around a NUMA-aware writer mutex and
per-node reader indicators, with a per-thread visible-reader fast path, admission
control, and spin-then-park waiting.

```cpp
#include <latch/latch.hpp>
#include <mutex>
#include <shared_mutex>

latch::RwLock lock;

void reader() { std::shared_lock guard(lock); /* ... */ }
void writer() { std::unique_lock guard(lock); /* ... */ }
```

`RwLock` meets the *SharedTimedMutex* requirements. Read locks may be nested, even
with a writer waiting. Other nesting is refused: `lock()` or `lock_shared()` by a thread
holding the write lock, or `lock()` by a thread holding a read lock, throws
`std::system_error` with `std::errc::resource_deadlock_would_occur`. Nothing is acquired
and what the thread held is unchanged, so this is safe through `std::unique_lock`,
`std::shared_lock` and `std::lock_guard`. The `try_` forms return `false` instead, and the
timed forms do so at once, as waiting cannot help. Code built with `-fno-exceptions` has
no way to carry these errors, so there the same calls abort.

For a write lock that a thread can take again, use `latch::RecursiveRwLock`, which is
`latch::BasicRwLock<latch::Recursion::write>`:

```cpp
latch::RecursiveRwLock lock;

void inner() { std::unique_lock guard(lock); /* ... */ }
void outer() { std::unique_lock guard(lock); inner(); }
```

Each `lock()` needs its own `unlock()`, and the lock stays exclusive until the last one.
Nesting past `latch::max_write_nesting` levels (1024) throws
`std::errc::resource_unavailable_try_again`. Mixing read and write locks still throws. Do not wait on a
`std::condition_variable_any` while nested, as it releases only one level. `RwLock` does
not contain the depth or its checks. Both are 8 bytes, as the depth uses spare bits of the
pointer.

By default a lock adjusts itself to the mix of reads and writes it observes. To fix its
behavior instead, give the expected share of writes as a percentage:

```cpp
latch::RwLock lock{latch::ExpectedWrites{50}};
```

This is only a performance hint, and a wrong value cannot affect correctness.

## Background

The design is based on [NUMA-Aware Reader-Writer Locks](https://cs.brown.edu/people/irina/papers/ppopp13.pdf)
(Calciu, Dice, Lev, Luchangco, Marathe and Shavit, PPoPP 2013).

Taken from the paper: the writer-preference lock (C-RW-WP), the C-PTL-TKT writer mutex
with its local hand-off, and per-node reader indicators split into ingress and egress
counters.

Where this implementation differs:

| Area | Paper | Here |
|---|---|---|
| Lock variants | Neutral, reader and writer preference | Writer preference only |
| Read path | Every reader updates its node's indicator | Fast path first: a reader publishes in its own row of a reader table while the lock is biased toward readers, so readers share no written cache line. The indicators are the slow path. The idea follows BRAVO (Dice and Kogan, USENIX ATC 2019). |
| Detecting a writer | Readers check whether the write mutex is locked | Readers check a mode flag, which stays set across a stream of writers so that they can skip draining the indicators |
| Reader starvation | A reader that runs out of patience raises a barrier that holds back new writers | Blocked readers elect a leader, which queues on the write mutex and admits the group. The last of the group holds the mutex until its own unlock. |
| Waiting | Spinning | Spin while the lock makes progress, then futex |
| More threads than CPUs | Not addressed | Queue depth is capped per node and surplus threads sleep at a gate, a form of concurrency restriction (Dice, Malthusian Locks, EuroSys 2017) |
| Tuning for the load | Fixed | Adjusts automatically to the observed write share, or set a fixed `ExpectedWrites` |
| Interface | `LD_PRELOAD` of `pthread_rwlock_t` | Header-only C++ type, with `LD_PRELOAD` as an option |

## Requirements

Linux, glibc, CMake 3.25+. Tested with GCC 14 and Clang 20. Tested on x86-64 only. aarch64 has
code paths but has not been built or run.

## Install

```sh
cmake -S . -B build-pkg -DLATCH_BUILD_BENCH=OFF -DLATCH_BUILD_TESTS=OFF -DLATCH_BUILD_INTERPOSER=OFF
cd build-pkg
cpack -G DEB   # liblatch-rwlock-dev_1.0.0_all.deb
cpack -G RPM   # latch-rwlock-devel-1.0.0-1.noarch.rpm (needs rpmbuild)
```

Both packages contain only the headers and CMake package files, and are architecture
independent. Then:

```cmake
find_package(latch-rwlock 1.0 REQUIRED)
target_link_libraries(app PRIVATE latch::rwlock)
```

Or use `add_subdirectory()`, or `cmake --install`. As a subdirectory it adds no install
rules unless you set `LATCH_INSTALL=ON`, and no packaging. To declare `latch::RwLock` in
a header without the full definition, include `latch/fwd.hpp`, not `class RwLock;`.

## Running existing binaries

`liblatch_interpose.so` is an optional, unpackaged build target. It replaces
`pthread_rwlock_*` in an unmodified program, and also `std::shared_mutex` and
`std::shared_timed_mutex` in libstdc++:

```sh
LD_PRELOAD=build/liblatch_interpose.so ./your-program
```

These locks are never recursive. A nested write returns `EDEADLK`, as does a read under a
write or a write under a read. The timed forms check their arguments first, even if the
lock is free, as glibc does.

Process-shared locks are not supported, and `pthread_rwlock_init` returns `ENOTSUP` for
them. A process-shared lock placed in shared memory without an attribute-carrying init
is not detected, so do not use the library with such programs.

## How it works

| Path | Mechanism |
|---|---|
| Read, fast | Publish the lock pointer in this thread's own row of a reader table, then recheck the bias flag. Readers share no written cache line, so reads scale with CPUs and do not degrade with threads. |
| Read, slow | Bias off: increment this NUMA node's ingress counter, check the mode, leave through the egress counter. Counters are sequestered per node. |
| Reader starvation | A reader that finds a writer in the way retracts. The first such reader becomes a leader, queues on the write mutex, and opens the lock for the group. The last of the group holds the mutex until its own unlock. |
| Write | NUMA-aware mutex (partitioned ticket lock over per-node ticket locks), with up to `RunBound` local hand-offs. Then drain the indicators and revoke the bias. A stream of writers skips the drain. |
| Bias revocation | Scan one column of the reader table. Fast reads then stay off for `BiasMultiplier` times the revocation cost, which bounds its share of write time. |
| Too many threads | Each node's queue is capped at `AdmissionLimit`. Surplus threads sleep at a gate, so the running set stays small and cache-warm. A thread returning to a short queue walks straight in, and every 256 releases a sleeper is admitted regardless, for long-term fairness. |
| Waiting | Spin only while the lock is making progress, then futex. Wake-ups use the ticket as a bitset, so a release wakes its successor rather than the whole queue. |

Spin counts are scaled by the measured cost of `PAUSE`, which varies widely across x86
generations. All metadata comes from an mmap-backed type-stable pool, so
the library cannot recurse into `malloc`, and an unlock may read lock words after
another thread has destroyed the lock.

### Costs

- Each lock takes 1,536 bytes with two NUMA nodes: 8 bytes in your object and the rest
  in the pool. Almost all of it is padding that keeps contended words apart.
- Fairness is long-term, not FIFO. Admission control trades short-term order for
  throughput, and timed and try acquisitions poll rather than queue.
- Mutable global state uses default visibility, so shared libraries linked at startup
  share one instance, even when built with `-fvisibility=hidden`. An object that hides these
  symbols with a linker version script gets its own instance and must not share locks
  with the rest of the process. The interposer is such an object, by design: it does
  not exchange lock objects with native users.
- Code loaded with `dlopen` is not supported. Such code may not share this state, in
  which case sharing a lock with it does not exclude, and the failure is silent. The thread
  pointer also uses the `initial-exec` TLS model, which can stop such code from loading.

## Tuning

Environment variables, read once at start-up:

| Variable | Default | Meaning |
|---|---|---|
| `AdmissionLimit` | 8 | Queue depth per node before threads sleep. 0 disables. |
| `ExpectedWrites` | automatic | 0-100 fixes every lock's profile from this write percentage. For programs using `LD_PRELOAD`, which cannot use the constructor. |
| `Patience` | 2000 | Nominal pauses without progress before parking. Negative spins forever. |
| `RunBound` | 100 | Local hand-offs before ceding to another node. |
| `Bias` | 1 | Enable the reader-table fast path. |
| `BiasMultiplier` | 9 | Fast reads stay off for this many times the revocation cost. |
| `NodePlacement` | 0 | Bind per-node state to its home node. Measured no benefit. |
| `NodeRestrictionMask` | -1 | `0` maps all CPUs to one node, to test how much NUMA awareness helps. |
| `Warn`, `VERBOSE` | 1, 0 | Diagnostics on stderr. |

Compile-time switches are in `include/latch/config.hpp`. The defaults were chosen by
measurement with `scripts/ablation.sh`.

## Cache line size

The cache line size is fixed at compile time. It comes from
`std::hardware_destructive_interference_size`, and `LATCH_CACHE_LINE_SIZE` overrides it,
as a macro or as a CMake cache variable.

One rule to know: because this is header-only, every part of a program that shares a lock
must compile with the same value. Mismatched settings across shared libraries would be a
real bug, and the compiler can't catch it.

The socket count itself is still found at startup, since that depends on the machine a
binary runs on rather than where it's built. So the grant slots are sized at runtime, but
the cache-line size isn't.

## Build, test, benchmark

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                               # native and interposed API
cmake -S . -B build-tsan -DLATCH_TSAN=ON && cmake --build build-tsan -j
build-tsan/test_rwlock native

build/rw_bench ALength=64 NCSLength=50 WCSLength=4 RCSLength=10 _Duration=3000 +96@5
LD_PRELOAD=build/liblatch_interpose.so build/rw_bench ... +384@5
build/rw_bench Lock=native ... +96@5                 # or Lock=recursive
scripts/bench_matrix.py --out results.csv glibc= latch=build/liblatch_interpose.so
```

`+96@5` means 96 threads writing 5% of the time. `VStress=10` varies concurrency every
10 ms to shake out exclusion and progress bugs; ignore its performance figures.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.
