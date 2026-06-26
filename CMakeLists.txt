cmake_minimum_required(VERSION 3.14)
project(kvcache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Core library (header-only for now, but kept as a target) ---
add_library(kvcache INTERFACE)
target_include_directories(kvcache INTERFACE ${CMAKE_SOURCE_DIR}/include)

# --- Demo executable ---
add_executable(demo src/main.cpp)
target_link_libraries(demo PRIVATE kvcache)

# --- Benchmark executable ---
add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE kvcache)

# --- Tests (GoogleTest fetched automatically) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(lru_tests tests/test_lru_cache.cpp)
target_link_libraries(lru_tests PRIVATE kvcache GTest::gtest_main)

add_executable(rate_tests tests/test_rate_limiter.cpp)
target_link_libraries(rate_tests PRIVATE kvcache GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lru_tests)
gtest_discover_tests(rate_tests)
