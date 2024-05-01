include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/qdldl/include")

# Tests are only run in Debug mode.
set(CMAKE_CXX_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")

file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.c")

file(GLOB tests "${PROJECT_SOURCE_DIR}/tests/*.cpp")
list(REMOVE_ITEM tests "${PROJECT_SOURCE_DIR}/tests/main.cpp")

foreach(file ${tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp")
  target_link_libraries("${name}" qcoslib gtest_main)
  add_test(NAME ${name} COMMAND "${name}")
endforeach()