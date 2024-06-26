cmake_minimum_required(VERSION 3.18)
project(qcos)

# If build type is not specified set to Release.
if(QCOS_BUILD_TYPE STREQUAL Debug)
    set(QCOS_BUILD_TYPE ${QCOS_BUILD_TYPE})
    add_compile_definitions(QCOS_DEBUG)
    set(CMAKE_C_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")
else()
    set(QCOS_BUILD_TYPE Release)
    set(CMAKE_C_FLAGS "-O3 -march=native -Werror -Wall -Wextra")
endif()
message(STATUS "Build Type: " ${QCOS_BUILD_TYPE})
message(STATUS "Build Flags: " ${CMAKE_C_FLAGS})

# Set integer size to 32 bits.
message(STATUS "Using 32 byte integers")
set(QDLDL_LONG OFF)

# Set floating points to double precision.
message(STATUS "Using double precision floating point")

# Add qdldl.
add_subdirectory(lib/qdldl)

# Add amd
add_subdirectory(lib/amd)

# Build qcos shared library.
add_library(qcos SHARED)
target_link_libraries(qcos qdldl amd m)
target_include_directories(qcos PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qcos PUBLIC lib/amd)
target_sources(qcos PRIVATE src/qcos_api.c src/input_validation.c src/linalg.c src/kkt.c src/utils.c src/cone.c src/qcos_error.c src/timer.c src/equilibration.c)

# Build qcos demo.
add_executable(qcos_demo examples/qcos_demo.c)
target_include_directories(qcos_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qcos_demo qcos)

# Test definitions.
if(ENABLE_TESTING)
    configure_file(CMakeLists.txt.in
            googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
            ${CMAKE_BINARY_DIR}/googletest-build)
    enable_testing()
    add_subdirectory(tests)
endif()
