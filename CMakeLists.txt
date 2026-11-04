cmake_minimum_required(VERSION 3.25)

project(latch-rwlock
  VERSION 1.0.0
  DESCRIPTION "Header-only NUMA-aware reader-writer lock for C++26"
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(LATCH_BUILD_BENCH "Build the rw_bench benchmark" ${PROJECT_IS_TOP_LEVEL})
option(LATCH_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(LATCH_BUILD_INTERPOSER "Build the LD_PRELOAD pthread_rwlock library (not packaged)" ${PROJECT_IS_TOP_LEVEL})
option(LATCH_TSAN "Instrument tests with ThreadSanitizer" OFF)

option(LATCH_INSTALL "Generate install rules and packages" ${PROJECT_IS_TOP_LEVEL})

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# ---- The library: headers only ------------------------------------------------

add_library(latch_rwlock INTERFACE)
add_library(latch::rwlock ALIAS latch_rwlock)
target_compile_features(latch_rwlock INTERFACE cxx_std_26)
# Header-only, so this reaches consumers through the target. Empty: use the compiler's
# value for the target. A consumer using the installed package sets the macro itself.
set(LATCH_CACHE_LINE_SIZE "" CACHE STRING "Cache line size in bytes; empty uses the compiler's value")
if(LATCH_CACHE_LINE_SIZE)
  target_compile_definitions(latch_rwlock INTERFACE LATCH_CACHE_LINE_SIZE=${LATCH_CACHE_LINE_SIZE})
endif()
target_link_libraries(latch_rwlock INTERFACE Threads::Threads)
target_include_directories(latch_rwlock INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(latch_rwlock PROPERTIES EXPORT_NAME rwlock)

# ---- Project targets ----------------------------------------------------------

set(latch_warnings -Wall -Wextra -Wno-unused-parameter)

# Intel's JCC erratum makes a loop's speed depend on where its branches fall: the same read path
# measured 180M or 243M ops/s depending only on unrelated code. Without this, comparisons between
# builds measure the alignment of the code and not the lock.
set(latch_stable_alignment "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(latch_stable_alignment -Wa,-mbranches-within-32B-boundaries)
endif()

if(LATCH_BUILD_BENCH)
  add_executable(rw_bench bench/rw_bench.cpp)
  target_link_libraries(rw_bench PRIVATE latch::rwlock)
  target_compile_options(rw_bench PRIVATE ${latch_warnings} -O3 -g ${latch_stable_alignment})
endif()

if(LATCH_BUILD_INTERPOSER)
  function(latch_add_interposer name order)
    # Must not depend on libstdc++: it loads into arbitrary binaries.
    add_library(${name} SHARED src/pthread_interpose.cpp)
    target_compile_definitions(${name} PRIVATE LATCH_DEFAULT_ORDER=${order})
    target_link_libraries(${name} PRIVATE latch::rwlock)
    target_compile_options(${name} PRIVATE ${latch_warnings} -O3 -g ${latch_stable_alignment}
      -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-strict-aliasing
      -fno-semantic-interposition -fno-plt $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-attributes>)
    target_link_options(${name} PRIVATE
      -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/latch.map -nodefaultlibs)
    target_link_libraries(${name} PRIVATE c gcc_s gcc)
    set_target_properties(${name} PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/latch.map)
  endfunction()
  latch_add_interposer(latch_interpose fifo)
  latch_add_interposer(latch_interpose_barging barging)
endif()

if(LATCH_BUILD_TESTS)
  enable_testing()
  # The same tests under each order, as RwLock follows the default.
  function(latch_add_test_binary name order)
    add_executable(${name} tests/test_rwlock.cpp)
    target_compile_definitions(${name} PRIVATE LATCH_DEFAULT_ORDER=${order})
    target_link_libraries(${name} PRIVATE latch::rwlock)
    target_compile_options(${name} PRIVATE ${latch_warnings})
    if(LATCH_TSAN)
      target_compile_options(${name} PRIVATE -fsanitize=thread -O1 -g)
      target_link_options(${name} PRIVATE -fsanitize=thread)
    endif()
  endfunction()
  latch_add_test_binary(test_rwlock fifo)
  latch_add_test_binary(test_rwlock_barging barging)
  add_test(NAME native COMMAND test_rwlock native)
  add_test(NAME native_barging COMMAND test_rwlock_barging native)
  if(LATCH_BUILD_INTERPOSER AND NOT LATCH_TSAN)
    add_test(NAME pthread COMMAND test_rwlock pthread)
    set_tests_properties(pthread PROPERTIES
      ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:latch_interpose>")
    add_test(NAME pthread_barging COMMAND test_rwlock_barging pthread)
    set_tests_properties(pthread_barging PROPERTIES
      ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:latch_interpose_barging>")
  endif()
endif()

if(NOT LATCH_INSTALL)
  return()
endif()

# ---- Install: headers and find_package() support ------------------------------

install(DIRECTORY include/latch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(TARGETS latch_rwlock EXPORT latch-rwlock-targets)

# Header-only: the package is architecture independent, so the config-version file
# must not record the pointer size of the machine that built it.
set(latch_cmake_dir ${CMAKE_INSTALL_DATADIR}/cmake/latch-rwlock)
install(EXPORT latch-rwlock-targets NAMESPACE latch:: DESTINATION ${latch_cmake_dir})
configure_package_config_file(cmake/latch-rwlock-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/latch-rwlock-config.cmake
  INSTALL_DESTINATION ${latch_cmake_dir})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/latch-rwlock-config-version.cmake
  COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/latch-rwlock-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/latch-rwlock-config-version.cmake
  DESTINATION ${latch_cmake_dir})

install(FILES LICENSE-MIT LICENSE-APACHE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/latch-rwlock)

# ---- Packages: cpack -G DEB, cpack -G RPM -------------------------------------

set(CPACK_PACKAGE_NAME "liblatch-rwlock-dev")
set(CPACK_PACKAGE_VENDOR "latch-rwlock")
set(CPACK_PACKAGE_CONTACT "sunbains <sunbains@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_GENERATOR "DEB;RPM")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "latch-rwlock")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "all")
set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6-dev")

set(CPACK_RPM_PACKAGE_NAME "latch-rwlock-devel")
set(CPACK_RPM_PACKAGE_ARCHITECTURE "noarch")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "MIT OR Apache-2.0")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_PACKAGE_REQUIRES "glibc-devel")
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
  /usr/share/cmake /usr/include)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
