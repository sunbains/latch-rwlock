// SPDX-License-Identifier: MIT OR Apache-2.0
// Forward declarations, for headers that only pass locks by reference.
#pragma once

namespace latch {

// Whether a thread holding the write lock may take it again.
enum class Recursion : bool { none, write };

template <Recursion recursion = Recursion::none>
class BasicRwLock;

using RwLock = BasicRwLock<>;
using RecursiveRwLock = BasicRwLock<Recursion::write>;

}  // namespace latch
