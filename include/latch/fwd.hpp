// SPDX-License-Identifier: MIT OR Apache-2.0
// Forward declarations, for headers that only pass locks by reference.
#pragma once

namespace latch {

// Whether a thread holding the write lock may take it again.
enum class Recursion : bool { none, write };

// How writers that find the lock busy are ordered. fifo: nodes take strict turns. barging:
// a thread may overtake a waiter on another node, for a bounded time. That is cheaper when
// threads are few or span sockets, as the lock moves less. See latch_mutex.hpp.
enum class Order : bool { fifo, barging };

// Lets a build run existing code on another order, as the tests and interposer do. Every
// part of a program that passes a RwLock between them must agree on it.
#ifndef LATCH_DEFAULT_ORDER
#define LATCH_DEFAULT_ORDER fifo
#endif
inline constexpr Order default_order = Order::LATCH_DEFAULT_ORDER;

template <Recursion recursion = Recursion::none, Order order = default_order>
class BasicRwLock;

using RwLock = BasicRwLock<>;
using RecursiveRwLock = BasicRwLock<Recursion::write>;
using BargingRwLock = BasicRwLock<Recursion::none, Order::barging>;
using RecursiveBargingRwLock = BasicRwLock<Recursion::write, Order::barging>;

}  // namespace latch
