# Python requirement.
find_package(Python COMPONENTS Interpreter)
if (NOT Python_FOUND)
  message(FATAL_ERROR "Python is required to generate data for tests.")
endif()

if(qoco_BUILD_TYPE STREQUAL Debug)
    set(qoco_BUILD_TYPE ${qoco_BUILD_TYPE})
    add_compile_definitions(qoco_DEBUG)
    set(CMAKE_CXX_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")
else()
    set(qoco_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O3 -march=native -Werror -Wall -Wextra")
endif()
message(STATUS "Test Build Type: " ${qoco_BUILD_TYPE})
message(STATUS "Test Build Flags: " ${CMAKE_C_FLAGS})

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
  target_link_libraries("${name}" qoco gtest_main)
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
  target_link_libraries("${name}" qoco gtest_main)
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
  target_link_libraries("${name}" qoco gtest_main)
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
  target_link_libraries("${name}" qoco gtest_main)
  add_test(NAME ${name} COMMAND "${name}")
endforeach()