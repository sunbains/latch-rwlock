// SPDX-License-Identifier: MIT OR Apache-2.0
// NUMA discovery. Prefers sysfs, because the "physical id" in /proc/cpuinfo is a
// socket number and is wrong under sub-NUMA clustering; falls back to it otherwise.
// All memory comes from mmap: this runs during LD_PRELOAD start-up and must not
// re-enter malloc.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "latch/config.hpp"
#include "latch/diagnostic.hpp"

namespace latch {

class Topology {
 public:
  [[nodiscard]] static Topology query(bool verbose) noexcept;

  int m_cpu_count = 0;      // CPUs discovered
  int m_cpu_capacity = 0;   // length of m_node_map and m_geo_map
  int m_node_count = 1;     // dense logical nodes, not folded
  int* m_node_map = nullptr;  // logical CPU id -> dense logical node id
  int* m_geo_map = nullptr;   // logical CPU id -> geographic index (APIC order)
  int* m_physical_node = nullptr;  // dense logical node id -> kernel node id
};

namespace detail {

struct IdAssociation {
  int m_physical_id;  // initial APIC id: socket-core-thread tuple
  int m_node_id;
  int m_logical_id;
};

[[nodiscard]] inline void* map_zeroed(std::size_t size) noexcept {
  void* region = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return region == MAP_FAILED ? nullptr : region;
}

// A pseudo-file snapshot in an mmap'd NUL-terminated buffer. Pseudo-files give
// short reads, so read until EOF; if the buffer fills, grow and start over.
class FileSnapshot {
 public:
  explicit FileSnapshot(const char* path) noexcept {
    for (std::size_t capacity = 64 * 1024;; capacity *= 2) {
      const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
      if (descriptor < 0) return;
      auto* buffer = static_cast<char*>(map_zeroed(capacity));
      if (buffer == nullptr) {
        close(descriptor);
        return;
      }
      std::size_t extent = 0;
      for (;;) {
        const ssize_t count = read(descriptor, buffer + extent, capacity - extent);
        if (count <= 0) break;
        extent += static_cast<std::size_t>(count);
      }
      close(descriptor);
      if (extent < capacity) {  // strict: guarantees a trailing NUL
        m_buffer = buffer;
        m_capacity = capacity;
        m_extent = extent;
        return;
      }
      munmap(buffer, capacity);
    }
  }
  FileSnapshot(const FileSnapshot&) = LATCH_DELETE("owns a mapping");
  FileSnapshot& operator=(const FileSnapshot&) = LATCH_DELETE("owns a mapping");
  ~FileSnapshot() {
    if (m_buffer != nullptr) munmap(m_buffer, m_capacity);
  }

  char* m_buffer = nullptr;
  std::size_t m_capacity = 0;
  std::size_t m_extent = 0;
};

// Line is expected to look like "key : value"; returns the value or nullptr.
[[nodiscard]] inline char* match_key(char* line, const char* key) noexcept {
  while (*line == *key && *line != 0) {
    ++line;
    ++key;
  }
  if (*key != 0) return nullptr;
  while (*line == ' ' || *line == '\t') ++line;
  if (*line != ':') return nullptr;
  ++line;
  while (*line == ' ' || *line == '\t') ++line;
  return line;
}

// Parse a kernel cpulist such as "0-23,48-71", marking node_map[cpu] = node.
inline int apply_cpu_list(const char* text, int node, int* node_map, int capacity) noexcept {
  int count = 0;
  while (*text != 0 && *text != '\n') {
    char* end = nullptr;
    const long first = std::strtol(text, &end, 10);
    if (end == text) break;
    long last = first;
    text = end;
    if (*text == '-') {
      ++text;
      last = std::strtol(text, &end, 10);
      text = end;
    }
    for (long cpu = first; cpu <= last; ++cpu) {
      if (cpu >= 0 && cpu < capacity) {
        node_map[cpu] = node;
        ++count;
      }
    }
    if (*text == ',') ++text;
  }
  return count;
}

// Enumerate /sys/devices/system/node. Node ids can be sparse and memory-only nodes
// have no CPUs: only nodes with CPUs get a dense logical id.
inline bool discover_from_sysfs(Topology& topology) noexcept {
  int logical_node = 0;
  int cpus_found = 0;
  for (int kernel_node = 0; kernel_node < 4096; ++kernel_node) {
    char path[96] = "/sys/devices/system/node/node";
    char* cursor = path + std::strlen(path);
    cursor = std::to_chars(cursor, path + sizeof(path), kernel_node).ptr;
    std::strcpy(cursor, "/cpulist");
    FileSnapshot list(path);
    if (list.m_buffer == nullptr) continue;
    const int count = apply_cpu_list(list.m_buffer, logical_node, topology.m_node_map,
                                     topology.m_cpu_capacity);
    if (count > 0) {
      topology.m_physical_node[logical_node] = kernel_node;
      ++logical_node;
      cpus_found += count;
    }
  }
  if (logical_node == 0) return false;
  topology.m_node_count = logical_node;
  topology.m_cpu_count = cpus_found;
  return true;
}

// Associates processor / physical id / initial apicid.
// Fills ids; also supplies the NUMA map when sysfs is unavailable.
inline int parse_cpuinfo(IdAssociation* ids, int capacity) noexcept {
  FileSnapshot info("/proc/cpuinfo");
  if (info.m_buffer == nullptr) return 0;
  int count = 0;
  int logical_id = -1;
  int node_id = -1;
  char* cursor = info.m_buffer;
  char* const end = info.m_buffer + info.m_extent;
  while (cursor < end) {
    char* line = cursor;
    char* eol = cursor;
    while (eol < end && *eol != '\n') ++eol;
    if (eol < end) *eol++ = 0;
    cursor = eol;
    if (char* value = match_key(line, "processor")) {
      logical_id = static_cast<int>(std::strtol(value, nullptr, 0));
    } else if (char* socket = match_key(line, "physical id")) {
      node_id = static_cast<int>(std::strtol(socket, nullptr, 0));
    } else if (char* apic = match_key(line, "initial apicid")) {
      if (logical_id >= 0 && logical_id < capacity && count < capacity) {
        ids[count++] = IdAssociation{static_cast<int>(std::strtol(apic, nullptr, 0)),
                                     std::max(node_id, 0), logical_id};
      }
      logical_id = -1;
      node_id = -1;
    }
  }
  return count;
}

}  // namespace detail

inline Topology Topology::query(bool verbose) noexcept {
  using namespace detail;
  Topology topology;
  const int configured = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  // CPUs can come and go; leave room so a late CPU id still indexes the map.
  const int capacity = std::max(configured, 1) + 1;
  topology.m_cpu_capacity = capacity;

  // Simple bump placement: never released, and every slot
  // defaults to 0 so an unexpected CPU id cannot yield a wild index.
  const std::size_t bytes = (sizeof(IdAssociation) + 2 * sizeof(int)) * capacity + 4096 * sizeof(int);
  auto* space = static_cast<char*>(map_zeroed(bytes));
  if (space == nullptr) {
    diagnostic("Error: could not map topology tables\n");
    _exit(1);
  }
  auto* ids = reinterpret_cast<IdAssociation*>(space);
  topology.m_geo_map = reinterpret_cast<int*>(ids + capacity);
  topology.m_node_map = topology.m_geo_map + capacity;
  topology.m_physical_node = topology.m_node_map + capacity;

  const bool have_sysfs = discover_from_sysfs(topology);
  const int described = parse_cpuinfo(ids, capacity);

  // Sort by APIC id so geographically close CPUs get numerically close indices.
  std::sort(ids, ids + described, [](const IdAssociation& left, const IdAssociation& right) {
    return left.m_physical_id < right.m_physical_id;
  });
  for (int index = 0; index < described; ++index) {
    topology.m_geo_map[ids[index].m_logical_id] = index;
  }
  if (described == 0) {  // no APIC ids (e.g. not x86): CPU ids are already geographic
    for (int cpu = 0; cpu < capacity; ++cpu) topology.m_geo_map[cpu] = cpu;
  }

  if (!have_sysfs) {
    int highest = 0;
    for (int index = 0; index < described; ++index) {
      topology.m_node_map[ids[index].m_logical_id] = ids[index].m_node_id;
      highest = std::max(highest, ids[index].m_node_id);
    }
    topology.m_node_count = highest + 1;
    topology.m_cpu_count = std::max(described, 1);
    for (int node = 0; node <= highest && node < 4096; ++node) topology.m_physical_node[node] = node;
  }

  if (verbose) {
    diagnostic("Topology: cpus=%d nodes=%d source=%s\n", topology.m_cpu_count,
               topology.m_node_count, have_sysfs ? "sysfs" : "cpuinfo");
  }
  return topology;
}

}  // namespace latch
