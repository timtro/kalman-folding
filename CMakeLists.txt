cmake_minimum_required (VERSION 2.8)

project(KalmanFold)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-7)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set (CMAKE_CXX_STANDARD 17)
set(common "-fno-omit-frame-pointer -pg")
set(profiling "-fno-omit-frame-pointer -pg")
set(warnings "-Wall -Wextra -Werror -pedantic")
set(optimizations_debug "-O2")
set(optimizations_release "-O3 -flto -ffast-math")
set(debug "-g")
set(perf_bench "-fno-omit-frame-pointer")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${common} ${debug} ${warnings} ${optimizations_debug}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${common} ${optimizations_release}")

# for clang++ use -fuse-ld=gold to avoid LD error "File format not recognized"
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    # set(sanitizer "-fsanitize=address")
    # set(sanitizer "-fsanitize=memory")
    # set(sanitizer "-fsanitize=memory -fsanitize-memory-track-origins")
    # set(sanitizer "-fsanitize=thread")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${sanitizer}")
  endif(CMAKE_BUILD_TYPE MATCHES DEBUG)
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

enable_testing()
add_subdirectory (test)

if( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  execute_process( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
endif()
