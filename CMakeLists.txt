cmake_minimum_required(VERSION 3.18)

set(QOCO_VERSION_MAJOR "0")
set(QOCO_VERSION_MINOR "1")
set(QOCO_VERSION_PATCH "0")
set(QOCO_VERSION "${QOCO_VERSION_MAJOR}.${QOCO_VERSION_MINOR}.${QOCO_VERSION_PATCH}")

# Project name
project(qoco VERSION ${QOCO_VERSION})
message(STATUS "Building QOCO v${QOCO_VERSION}")

# If build type is not specified set to Release.
if(QOCO_BUILD_TYPE STREQUAL Debug)
    set(QOCO_BUILD_TYPE ${QOCO_BUILD_TYPE})
    add_compile_definitions(QOCO_DEBUG)
    set(CMAKE_C_FLAGS "-g -Wall -Wextra -fsanitize=address,undefined")
else()
    set(QOCO_BUILD_TYPE Release)
    set(CMAKE_C_FLAGS "-O3 -Wall -Wextra")
endif()
message(STATUS "Build Type: " ${QOCO_BUILD_TYPE})
message(STATUS "Build Flags: " ${CMAKE_C_FLAGS})

# Set integer size to 32 bits.
message(STATUS "Using 32 byte integers")
set(QDLDL_LONG OFF)

# Set floating points to double precision.
message(STATUS "Using double precision floating point")

# Detect operating system.
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_compile_definitions(IS_LINUX)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_compile_definitions(IS_MACOS)
endif()

# Add qdldl.
add_subdirectory(lib/qdldl)

# Add amd
add_subdirectory(lib/amd)

# Build qoco shared library.
add_library(qoco SHARED)
target_link_libraries(qoco qdldl amd m)
target_include_directories(qoco PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qoco PUBLIC lib/amd)
target_sources(qoco PRIVATE src/qoco_api.c src/input_validation.c src/linalg.c src/kkt.c src/utils.c src/cone.c src/qoco_error.c src/equilibration.c)
target_compile_definitions(qoco PUBLIC "QOCO_VERSION_MAJOR=${QOCO_VERSION_MAJOR}")
target_compile_definitions(qoco PUBLIC "QOCO_VERSION_MINOR=${QOCO_VERSION_MINOR}")
target_compile_definitions(qoco PUBLIC "QOCO_VERSION_PATCH=${QOCO_VERSION_PATCH}")

# Build qoco static library.
add_library(qocostatic STATIC)
target_link_libraries(qocostatic qdldlstatic amd m)
target_include_directories(qocostatic PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qocostatic PUBLIC lib/amd)
target_sources(qocostatic PRIVATE src/qoco_api.c src/input_validation.c src/linalg.c src/kkt.c src/utils.c src/cone.c src/qoco_error.c src/equilibration.c)
target_compile_definitions(qocostatic PUBLIC "QOCO_VERSION_MAJOR=${QOCO_VERSION_MAJOR}")
target_compile_definitions(qocostatic PUBLIC "QOCO_VERSION_MINOR=${QOCO_VERSION_MINOR}")
target_compile_definitions(qocostatic PUBLIC "QOCO_VERSION_PATCH=${QOCO_VERSION_PATCH}")

# Include timer file depending on OS.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_sources(qoco PRIVATE src/timer_linux.c)
    target_sources(qocostatic PRIVATE src/timer_linux.c)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    target_sources(qoco PRIVATE src/timer_macos.c)
    target_sources(qocostatic PRIVATE src/timer_macos.c)
endif()

# Build qoco demo.
add_executable(qoco_demo examples/qoco_demo.c)
target_include_directories(qoco_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qoco_demo qoco)

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

install(
    TARGETS qocostatic
    EXPORT qocostatic
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
    TARGETS qoco
    EXPORT qoco
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(FILES ${qoco_headers}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/qoco"
)
