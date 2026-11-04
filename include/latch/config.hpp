// SPDX-License-Identifier: MIT OR Apache-2.0
// Compile-time feature switches. Each technique can be turned off independently
// so its contribution can be measured. Defaults are selected by measurement.
#pragma once

#if __cplusplus <= 202302L
#error "latch requires C++26 (-std=c++26)"
#endif

#ifndef LATCH_BIAS
#define LATCH_BIAS 1              // per-thread visible-reader slots (BRAVO family)
#endif
#ifndef LATCH_ASYMMETRIC
#define LATCH_ASYMMETRIC 0        // membarrier() instead of a reader-side fence: 8% faster
                                   // pure reads, but 45% slower at 1% writes on 96 CPUs
#endif
#ifndef LATCH_PARKING
#define LATCH_PARKING 1           // spin-then-futex waiting
#endif
#ifndef LATCH_EAGER_REOPEN
#define LATCH_EAGER_REOPEN 0      // last writer reopens the reader fast path
#endif
#ifndef LATCH_PROPORTIONAL_TICKET
#define LATCH_PROPORTIONAL_TICKET 0  // node ticket wait scales with queue distance
#endif
#ifndef LATCH_PREFETCHW
#define LATCH_PREFETCHW 1         // avoid the RTS->RTO upgrade before CAS
#endif
#ifndef LATCH_SPLIT_INDICATOR
#define LATCH_SPLIT_INDICATOR 0   // ingress and egress on separate coherence units
#endif
#ifndef LATCH_CALIBRATE_PAUSE
#define LATCH_CALIBRATE_PAUSE 1   // scale spin counts by the measured cost of PAUSE
#endif
#ifndef LATCH_FETCH_POLICY
#define LATCH_FETCH_POLICY 3      // 0=backoff 1=tdma 2=numa 3=xadd
#endif
#ifndef LATCH_CPU_SOURCE
#define LATCH_CPU_SOURCE 0        // 0=rseq 1=sched_getcpu 2=rdtscp
#endif
#ifndef LATCH_CLAIMS
#define LATCH_CLAIMS 0            // run-time assertions
#endif

// C++26 deleted functions with a reason, where the compiler has them yet.
#if defined(__cpp_deleted_function) && __cpp_deleted_function >= 202403L
#define LATCH_DELETE(reason) delete (reason)
#else
#define LATCH_DELETE(reason) delete
#endif

namespace latch {

enum class FetchIncrementPolicy { backoff, tdma, numa, xadd };

inline constexpr bool bias_compiled = LATCH_BIAS != 0;
inline constexpr bool asymmetric_compiled = LATCH_ASYMMETRIC != 0;
inline constexpr bool parking_compiled = LATCH_PARKING != 0;
inline constexpr bool eager_reopen_compiled = LATCH_EAGER_REOPEN != 0;
inline constexpr bool proportional_ticket_compiled = LATCH_PROPORTIONAL_TICKET != 0;
inline constexpr bool prefetchw_compiled = LATCH_PREFETCHW != 0;
inline constexpr bool split_indicator_compiled = LATCH_SPLIT_INDICATOR != 0;
inline constexpr bool calibrate_pause_compiled = LATCH_CALIBRATE_PAUSE != 0;
inline constexpr bool claims_compiled = LATCH_CLAIMS != 0;
inline constexpr auto fetch_policy = static_cast<FetchIncrementPolicy>(LATCH_FETCH_POLICY);

}  // namespace latch
