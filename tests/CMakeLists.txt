cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 14)

# Python requirement.
find_package(Python COMPONENTS Interpreter)

if(NOT Python_FOUND)
  message(FATAL_ERROR "Python is required to generate data for tests.")
endif()

if(QOCO_BUILD_TYPE STREQUAL Debug)
  set(QOCO_BUILD_TYPE ${QOCO_BUILD_TYPE})
  add_compile_definitions(QOCO_DEBUG)

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_FLAGS "-g -Wall")
  else()
    set(CMAKE_CXX_FLAGS "-g -march=native -Wall")
  endif()

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
  endif()
else()
  set(QOCO_BUILD_TYPE Release)

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_FLAGS "-O3 -Wall")
  else()
    set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall")
  endif()
endif()

# Add -Wextra if not on Windows.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

# Enable exception handling on Windows.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(MSVC)
    # Only add MSVC-style flags when using MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif()
endif()

message(STATUS "Test Build Type: " ${QOCO_BUILD_TYPE})
message(STATUS "Test Build Flags: " ${CMAKE_CXX_FLAGS})

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/tests")
include_directories("${PROJECT_SOURCE_DIR}/tests/utils")
include_directories("${PROJECT_SOURCE_DIR}/lib/qdldl/include")

# Unit tests.
file(GLOB unit_tests "${PROJECT_SOURCE_DIR}/tests/unit_tests/*.cpp")

foreach(file ${unit_tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "unit_tests/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/utils/test_utils.cpp")
  target_link_libraries("${name}" qocostatic gtest_main)
  # Link CUDA libraries for CUDA backend
  if(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    get_property(qoco_cuda_libs GLOBAL PROPERTY QOCO_CUDA_LIBS)
    target_link_libraries("${name}" ${qoco_cuda_libs})
  endif()
  add_test(NAME ${name} COMMAND "${name}")
endforeach()

# Simple convex tests.
file(GLOB simple_tests "${PROJECT_SOURCE_DIR}/tests/simple_tests/*.cpp")

foreach(file ${simple_tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "simple_tests/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/utils/test_utils.cpp")
  target_link_libraries("${name}" qocostatic gtest_main)
  # Link CUDA libraries for CUDA backend
  if(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    get_property(qoco_cuda_libs GLOBAL PROPERTY QOCO_CUDA_LIBS)
    target_link_libraries("${name}" ${qoco_cuda_libs})
  endif()
  add_test(NAME ${name} COMMAND "${name}")
endforeach()

# Generate problem data.
execute_process(COMMAND python generate_all_problem_data.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

file(GLOB ocp "${PROJECT_SOURCE_DIR}/tests/ocp/*.cpp")

foreach(file ${ocp})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "ocp/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/utils/test_utils.cpp")
  target_link_libraries("${name}" qocostatic gtest_main)
  # Link CUDA libraries for CUDA backend
  if(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    get_property(qoco_cuda_libs GLOBAL PROPERTY QOCO_CUDA_LIBS)
    target_link_libraries("${name}" ${qoco_cuda_libs})
  endif()
  add_test(NAME ${name} COMMAND "${name}")
endforeach()

file(GLOB portfolio "${PROJECT_SOURCE_DIR}/tests/portfolio/*.cpp")

foreach(file ${portfolio})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "portfolio/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/utils/test_utils.cpp")
  target_link_libraries("${name}" qocostatic gtest_main)
  # Link CUDA libraries for CUDA backend
  if(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    get_property(qoco_cuda_libs GLOBAL PROPERTY QOCO_CUDA_LIBS)
    target_link_libraries("${name}" ${qoco_cuda_libs})
  endif()
  add_test(NAME ${name} COMMAND "${name}")
endforeach()