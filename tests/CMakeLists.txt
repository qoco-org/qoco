include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/tests")
include_directories("${PROJECT_SOURCE_DIR}/lib/qdldl/include")

# Tests are only run in Debug mode.
set(CMAKE_CXX_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")
file(GLOB convex_tests "${PROJECT_SOURCE_DIR}/tests/convex_tests/*.cpp")
file(GLOB unit_tests "${PROJECT_SOURCE_DIR}/tests/unit_tests/*.cpp")

foreach(file ${convex_tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "convex_tests/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp")
  target_link_libraries("${name}" qcoslib gtest_main)
  add_test(NAME ${name} COMMAND "${name}")
endforeach()

foreach(file ${unit_tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "unit_tests/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/test_utils.cpp")
  target_link_libraries("${name}" qcoslib gtest_main)
  add_test(NAME ${name} COMMAND "${name}")
endforeach()