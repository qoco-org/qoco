cmake_minimum_required(VERSION 3.18)

set(QOCO_VERSION_MAJOR "0")
set(QOCO_VERSION_MINOR "1")
set(QOCO_VERSION_PATCH "5")
set(QOCO_VERSION "${QOCO_VERSION_MAJOR}.${QOCO_VERSION_MINOR}.${QOCO_VERSION_PATCH}")

# Project name
project(qoco VERSION ${QOCO_VERSION})
message(STATUS "Building QOCO v${QOCO_VERSION}")

# If build type is not specified set to Release.
if(QOCO_BUILD_TYPE STREQUAL Debug)
    set(QOCO_BUILD_TYPE ${QOCO_BUILD_TYPE})
    add_compile_definitions(QOCO_DEBUG)
    set(CMAKE_C_FLAGS "-g -Wall")
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address,undefined")
    endif()
else()
    set(QOCO_BUILD_TYPE Release)
    set(CMAKE_C_FLAGS "-O3 -Wall")
endif()

# Detect operating system.
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(IS_LINUX 1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(IS_MACOS 1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(IS_WINDOWS 1)
    if(QOCO_BUILD_TYPE STREQUAL Release)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox")
    endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/configure/qoco_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/qoco_config.h @ONLY)

message(STATUS "Build Type: " ${QOCO_BUILD_TYPE})
message(STATUS "Build Flags: " ${CMAKE_C_FLAGS})

# Set integer size to 32 bits.
message(STATUS "Using 32 byte integers")
set(QDLDL_LONG OFF)

# Set floating points to double precision.
message(STATUS "Using double precision floating point")


# Add -fPIC option explicity.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add qdldl.
add_subdirectory(lib/qdldl)

# Add amd
add_subdirectory(lib/amd)

# Are we building for matlab. If so we will define printf as mexPrintf to print into matlab terminal.
if(${MATLAB})
    add_compile_definitions(MATLAB)
endif()

set(qoco_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/qoco_api.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/input_validation.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/qoco_linalg.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/kkt.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cone.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/qoco_status.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/equilibration.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/qoco_utils.c"
)

set(qoco_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/qoco.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/qoco_api.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/input_validation.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/qoco_linalg.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/kkt.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cone.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/qoco_status.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/equilibration.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/enums.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/definitions.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/structs.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/timer.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/qoco_utils.h"
)

# Include timer file depending on OS.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND qoco_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/timer_linux.c")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    list(APPEND qoco_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/timer_macos.c")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(APPEND qoco_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/timer_windows.c")
endif()

# Build qoco shared library.
add_library(qoco SHARED)
target_link_libraries(qoco qdldl amd)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    target_link_libraries(qoco m)
endif()
target_include_directories(qoco PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qoco PUBLIC lib/amd)
target_include_directories(qoco PUBLIC lib/qdldl/include)
target_sources(qoco PRIVATE ${qoco_sources})

# Build qoco static library.
add_library(qocostatic STATIC)
target_link_libraries(qocostatic qdldlstatic amd)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    target_link_libraries(qocostatic m)
endif()
target_include_directories(qocostatic PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qocostatic PUBLIC lib/amd)
target_include_directories(qocostatic PUBLIC lib/qdldl/include)
target_sources(qocostatic PRIVATE ${qoco_sources})

# Build qoco demo.
if(BUILD_QOCO_DEMO)
    add_executable(qoco_demo ${PROJECT_SOURCE_DIR}/examples/qoco_demo.c)
    target_include_directories(qoco_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(qoco_demo qocostatic)
endif()

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
