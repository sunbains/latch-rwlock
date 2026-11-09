# latch-rwlock

A header-only, NUMA-aware reader-writer lock for C++23 on Linux, designed to keep
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

### Choosing how writers are ordered

Writers that find the lock busy are ordered by `latch::Order`. The default, `fifo`, gives NUMA
nodes strict turns. With `barging`, a thread can overtake a waiter on another node, for a
bounded number of acquisitions. That moves the lock between sockets less often:

```cpp
latch::BargingRwLock lock;           // latch::BasicRwLock<latch::Recursion::none, latch::Order::barging>
latch::RecursiveBargingRwLock other; // latch::BasicRwLock<latch::Recursion::write, latch::Order::barging>
```

See [Performance](#performance) for when each is the better choice.

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
LD_PRELOAD=build/liblatch_interpose_barging.so ./your-program   # with Order::barging
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
| Write, FIFO | NUMA-aware mutex (partitioned ticket lock over per-node ticket locks), with up to `RunBound` local hand-offs. Then drain the indicators and revoke the bias. A stream of writers skips the drain. |
| Write, barging | The top level is one word instead, so an uncontended acquisition takes one atomic rather than two. An arrival may take a free word if its own node's queue is empty. The head of each node's queue polls slowly while another node keeps re-acquiring, which lets that node run a batch, and takes a lock that has gone idle. Whoever holds the lock counts acquisitions, and after `BargeLimit` of them gives it to the next waiting node, round-robin. Draining is as for FIFO. |
| Bias revocation | Scan one column of the reader table. Fast reads then stay off for `BiasMultiplier` times the revocation cost, which bounds its share of write time. |
| Too many threads | Each node's queue is capped at `AdmissionLimit`. Surplus threads sleep at a gate, so the running set stays small and cache-warm. A thread returning to a short queue walks straight in, and every 256 releases a sleeper is admitted regardless, for long-term fairness. |
| Waiting | Spin only while the lock is making progress, then futex. Wake-ups use the ticket as a bitset, so a release wakes its successor rather than the whole queue. Barging's waiters spin for longer, as they expect to sit out other nodes' batches. |

Spin counts are scaled by the measured cost of `PAUSE`, which varies widely across x86
generations. All metadata comes from an mmap-backed type-stable pool, so
the library cannot recurse into `malloc`, and an unlock may read lock words after
another thread has destroyed the lock.

### Costs

- Each lock takes 1,536 bytes with two NUMA nodes, or 1,152 with `Order::barging`: 8 bytes in
  your object and the rest in the pool. Almost all of it is padding that keeps contended words
  apart.
- Fairness is long-term, whichever the `Order`. Admission control trades short-term order for
  throughput. With `Order::barging`, a queue head waits about `(nodes - 1) * BargeLimit`
  acquisitions once it is registered.
- Timed and try acquisitions poll rather than queue. On a node whose queue never empties, a
  timed acquisition can run to its deadline and fail, however long that is.
- Mutable global state uses default visibility, so shared libraries linked at startup
  share one instance, even when built with `-fvisibility=hidden`. An object that hides these
  symbols with a linker version script gets its own instance and must not share locks
  with the rest of the process. The interposer is such an object, by design: it does
  not exchange lock objects with native users.
- Code loaded with `dlopen` is not supported. Such code may not share this state, in
  which case sharing a lock with it does not exclude, and the failure is silent. The thread
  pointer also uses the `initial-exec` TLS model, which can stop such code from loading.

## Performance

Millions of operations per second, through `LD_PRELOAD`, so all three run the same benchmark
binary. Higher is better and the best of each row is in bold. "Barging / FIFO" above 1.00 means
barging is faster.

### Test system

| | |
|---|---|
| CPU | 2 x Intel Xeon Platinum 8260 (Cascade Lake), 2.4 GHz base, 3.9 GHz max |
| Topology | 96 CPUs: 2 sockets, 24 cores each, 2 threads per core, 2 NUMA nodes |
| Cache | 48 MiB L2 in total, 71.5 MiB L3 in total |
| Memory | 123 GiB |
| OS | Ubuntu 25.04 userland, Linux 6.14, bare metal, `performance` governor, turbo on |
| Software | GCC 14.2 Release build, glibc 2.41 |

### Workload

```sh
scripts/bench_matrix.py --threads 1,2,3,4,6,8,24,48,96,192,384 --writes 0,1,5,50,100 \
  --reps 3 --duration 1000 glibc= fifo=build/liblatch_interpose.so \
  barging=build/liblatch_interpose_barging.so
```

Each thread loops over a critical section on one lock and non-critical work on private data
(`ALength=64 NCSLength=50 WCSLength=4 RCSLength=10`). A write touches 8 shared words, a read
20, and the work between them 100. Each figure is the median of 3 runs of 1 second, with the
locks taking turns so that drift affects all of them alike. Every run passed the benchmark's
own check that the protected data stayed consistent.

### Threads placed by the scheduler

**0% writes**

| Threads | glibc | FIFO | Barging | Barging / FIFO |
|---:|---:|---:|---:|---:|
| 1 | 4.48 | **4.50** | 4.48 | 0.99 |
| 2 | 5.61 | **8.94** | 8.92 | 1.00 |
| 3 | 4.81 | **13.45** | 13.41 | 1.00 |
| 4 | 4.75 | **17.31** | 17.24 | 1.00 |
| 6 | 4.85 | 25.57 | **25.69** | 1.00 |
| 8 | 5.08 | **33.65** | 33.64 | 1.00 |
| 24 | 4.82 | 99.81 | **99.91** | 1.00 |
| 48 | 5.19 | 171.9 | **172.0** | 1.00 |
| 96 | 8.03 | 231.1 | **238.3** | 1.03 |
| 192 | 4.86 | **236.7** | 235.6 | 1.00 |
| 384 | 7.15 | **232.9** | 230.4 | 0.99 |

**1% writes**

| Threads | glibc | FIFO | Barging | Barging / FIFO |
|---:|---:|---:|---:|---:|
| 1 | 4.47 | 4.48 | **4.48** | 1.00 |
| 2 | 5.54 | 7.18 | **8.54** | 1.19 |
| 3 | 4.59 | **8.84** | 8.42 | 0.95 |
| 4 | 4.25 | **10.39** | 9.70 | 0.93 |
| 6 | 3.44 | **11.40** | 10.90 | 0.96 |
| 8 | 3.46 | **12.99** | 11.90 | 0.92 |
| 24 | 3.25 | **19.36** | 18.82 | 0.97 |
| 48 | 2.76 | 27.22 | **31.28** | 1.15 |
| 96 | 3.24 | **42.57** | 31.87 | 0.75 |
| 192 | 2.56 | **41.05** | 34.75 | 0.85 |
| 384 | 3.32 | **45.32** | 37.95 | 0.84 |

**5% writes**

| Threads | glibc | FIFO | Barging | Barging / FIFO |
|---:|---:|---:|---:|---:|
| 1 | **4.45** | 4.44 | 4.44 | 1.00 |
| 2 | 2.54 | 7.09 | **7.36** | 1.04 |
| 3 | 2.44 | **5.26** | 5.20 | 0.99 |
| 4 | 2.15 | 5.44 | **6.11** | 1.12 |
| 6 | 1.79 | **5.86** | 5.82 | 0.99 |
| 8 | 1.78 | **6.13** | 5.96 | 0.97 |
| 24 | 1.76 | **6.73** | 6.64 | 0.99 |
| 48 | 1.65 | 11.12 | **12.24** | 1.10 |
| 96 | 1.73 | **15.62** | 11.12 | 0.71 |
| 192 | 1.53 | **11.16** | 10.78 | 0.97 |
| 384 | 1.73 | **13.64** | 11.23 | 0.82 |

**50% writes**

| Threads | glibc | FIFO | Barging | Barging / FIFO |
|---:|---:|---:|---:|---:|
| 1 | **4.47** | 4.31 | 4.38 | 1.02 |
| 2 | 2.45 | 1.47 | **3.70** | 2.52 |
| 3 | 1.34 | 1.88 | **3.54** | 1.89 |
| 4 | 1.26 | 2.54 | **3.23** | 1.27 |
| 6 | 1.18 | 2.92 | **3.03** | 1.04 |
| 8 | 1.16 | **3.17** | 3.02 | 0.95 |
| 24 | 1.08 | **3.05** | 2.75 | 0.90 |
| 48 | 0.97 | **2.72** | 2.43 | 0.89 |
| 96 | 1.02 | **2.38** | 2.03 | 0.85 |
| 192 | 1.05 | 1.75 | **1.86** | 1.06 |
| 384 | 1.08 | **1.94** | 1.85 | 0.95 |

**100% writes**

| Threads | glibc | FIFO | Barging | Barging / FIFO |
|---:|---:|---:|---:|---:|
| 1 | **4.83** | 4.65 | 4.72 | 1.01 |
| 2 | 3.04 | 3.33 | **3.45** | 1.04 |
| 3 | 2.43 | 2.09 | **3.66** | 1.75 |
| 4 | 1.89 | 3.06 | **3.48** | 1.14 |
| 6 | 1.54 | **2.88** | 2.87 | 1.00 |
| 8 | 1.53 | **2.86** | 2.61 | 0.91 |
| 24 | 1.43 | **1.97** | 1.91 | 0.97 |
| 48 | 1.35 | **1.94** | 1.88 | 0.97 |
| 96 | 1.36 | **2.03** | 1.99 | 0.98 |
| 192 | 0.25 | **2.01** | 1.97 | 0.98 |
| 384 | 0.51 | **2.02** | 1.94 | 0.96 |

### Threads pinned to chosen CPUs

Where threads run decides much of the result, so this removes that variable.

| Placement | Writes | glibc | FIFO | Barging | Barging / FIFO |
|---|---:|---:|---:|---:|---:|
| 1 thread | 5% | **4.48** | 4.42 | 4.43 | 1.00 |
| 1 thread | 50% | **4.44** | 4.12 | 4.21 | 1.02 |
| 1 thread | 100% | **4.83** | 4.61 | 4.80 | 1.04 |
| 2, one socket | 5% | 4.65 | 6.72 | **6.93** | 1.03 |
| 2, one socket | 50% | 2.32 | 3.06 | **3.32** | 1.08 |
| 2, one socket | 100% | 2.87 | 3.10 | **3.51** | 1.13 |
| 2, hyperthread siblings | 5% | **6.17** | 6.13 | 6.13 | 1.00 |
| 2, hyperthread siblings | 50% | **5.95** | 5.59 | 5.69 | 1.02 |
| 2, hyperthread siblings | 100% | **6.74** | 6.49 | 6.60 | 1.02 |
| 2, across sockets | 5% | 2.39 | **4.13** | 4.02 | 0.97 |
| 2, across sockets | 50% | 1.21 | 1.34 | **3.66** | 2.74 |
| 2, across sockets | 100% | 2.46 | 1.60 | **4.29** | 2.68 |
| 3, 2+1 across sockets | 5% | 2.35 | **4.17** | 4.11 | 0.99 |
| 3, 2+1 across sockets | 50% | 1.41 | 1.44 | **3.18** | 2.20 |
| 3, 2+1 across sockets | 100% | 2.26 | 1.97 | **3.47** | 1.76 |
| 4, 2+2 across sockets | 5% | 1.94 | **5.06** | 4.85 | 0.96 |
| 4, 2+2 across sockets | 50% | 1.24 | 2.16 | **3.22** | 1.49 |
| 4, 2+2 across sockets | 100% | 1.75 | 2.10 | **3.02** | 1.43 |
| 8, 4+4 across sockets | 5% | 1.78 | **6.02** | 5.96 | 0.99 |
| 8, 4+4 across sockets | 50% | 1.17 | 2.92 | **2.96** | 1.01 |
| 8, 4+4 across sockets | 100% | 1.53 | **2.78** | 2.58 | 0.93 |

### Uncontended cost

One thread locking and unlocking in a tight loop on one CPU, with no other work: the best of
5 runs, in nanoseconds per pair. glibc's uncontended read path is the cheapest of the three.

| Pair | glibc | FIFO | Barging |
|---|---:|---:|---:|
| Write lock and unlock | 23.6 | 23.3 | **15.1** |
| Read lock and unlock | **15.6** | 21.0 | 19.7 |

### Reading the results

- **Reads scale, and glibc's do not.** With no writes at 96 threads and above, latch reaches
  230M or more against glibc's 8M or less, as readers share no written cache line. Past 96
  threads there are no more CPUs to use, so throughput stays flat.
- **FIFO's weak spot is few writers across sockets.** With nobody on its own node to pass the
  lock to, every release crosses the socket link. 2 threads across sockets at 100% writes do
  1.60M, behind glibc's 2.46M. Barging lets one node run a batch and does 4.29M.
- **Barging's weak spot is many threads with few writes.** At 96 threads it is 0.75x FIFO at 1%
  writes and 0.71x at 5%. At 8 threads with 1% writes it is 0.92x. The cause is not yet known.
  Its overall ratio to FIFO across this table is 1.02x, and 1.09x at 8 threads or fewer.
- **A few cases still favor glibc.** One thread with many writes, where it is up to 5% ahead
  of barging and 7% ahead of FIFO when pinned, and hyperthread siblings, up to 6% ahead.

Choose by what your threads do. FIFO is the default and the more even at high thread counts.
Barging suits a few threads doing frequent writes, especially when they span sockets.

### Caveats

- **This is one workload on one machine.** Critical sections here are very short, so the lock
  is a large share of the work. Longer ones should narrow these gaps, which has not been
  measured. Results on other CPUs, or with one socket, may differ.
- **Unpinned results depend on scheduling.** The same setup can land differently from run to
  run. For example, FIFO with 2 threads at 50% writes ranged from about 1.4M to about 3.2M,
  apparently depending on whether they shared a socket, so that row's median is not a typical
  figure.
- **Three short runs resolve large differences, not small ones.** Treat a few percent as noise.
- **Throughput is not fairness.** A fixed-time benchmark rewards a lock that favors some
  threads. Jain's fairness index, where 1.00 is perfectly even, was 0.95 or better in 145 of
  the 165 unpinned results. The exceptions: at 384 threads it falls to 0.68 to 0.81 on all
  three with no writes, and to 0.89 to 0.93 on FIFO and barging with 100% writes. FIFO and
  barging fall to 0.70 to 0.94 at 3 to 8 threads with 100% writes, and FIFO to 0.91 at 6 to 8
  threads with 50% writes. glibc falls to 0.84 to 0.93 with no or 1% writes at 3 to 8 threads.
  Barging's turns go round nodes, not threads, so a thread alone on its node gets more than one
  sharing it: 3 threads pinned 2+1 show 0.89.
- **Tail latency was not measured here.** These figures say nothing about the slowest
  acquisition of a run.
- **The build affects the numbers.** Intel's JCC erratum makes a loop's speed depend on where
  its branches fall: one read path measured 180M or 243M from unrelated changes. The benchmark
  and interposers build with `-Wa,-mbranches-within-32B-boundaries`, so that comparisons do not
  measure this. A program using the headers gets it only if built with that flag.

## Tuning

Environment variables, read once at start-up:

| Variable | Default | Meaning |
|---|---|---|
| `AdmissionLimit` | 8 | Queue depth per node before threads sleep. 0 disables. |
| `ExpectedWrites` | automatic | 0-100 fixes every lock's profile from this write percentage. For programs using `LD_PRELOAD`, which cannot use the constructor. |
| `Patience` | 2000 | Nominal pauses without progress before parking. Negative spins forever. |
| `RunBound` | 100 | Local hand-offs before ceding to another node. |
| `BargeLimit` | 256 | `Order::barging`: acquisitions before a waiting node gets its turn. Lower is fairer in the short term and moves the lock more. |
| `RemotePoll` | 1024 | `Order::barging`: nominal pauses between a queue head's looks at a lock another node keeps re-acquiring. |
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
binary runs on rather than where it's built. So `FifoMutex`'s grant slots are sized at runtime,
but the cache-line size isn't.

## Build, test, benchmark

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                               # native and interposed API, in both orders
cmake -S . -B build-tsan -DLATCH_TSAN=ON && cmake --build build-tsan -j
build-tsan/test_rwlock native && build-tsan/test_rwlock_barging native

build/rw_bench ALength=64 NCSLength=50 WCSLength=4 RCSLength=10 _Duration=3000 +96@5
LD_PRELOAD=build/liblatch_interpose.so build/rw_bench ... +384@5
build/rw_bench Lock=native ... +96@5                 # or Lock=recursive, Lock=barging
scripts/bench_matrix.py --out results.csv glibc= latch=build/liblatch_interpose.so
```

`+96@5` means 96 threads writing 5% of the time. `VStress=10` varies concurrency every
10 ms to shake out exclusion and progress bugs; ignore its performance figures.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.
