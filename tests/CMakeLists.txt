include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/tests")
include_directories("${PROJECT_SOURCE_DIR}/lib/qdldl/include")

if(QCOS_BUILD_TYPE STREQUAL Debug)
    set(QCOS_BUILD_TYPE ${QCOS_BUILD_TYPE})
    add_compile_definitions(QCOS_DEBUG)
    set(CMAKE_CXX_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")
else()
    set(QCOS_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O3 -march=native -Werror -Wall -Wextra")
endif()

message(STATUS "Test Build Type: " ${QCOS_BUILD_TYPE})
message(STATUS "Test Build Flags: " ${CMAKE_C_FLAGS})

file(GLOB convex_tests "${PROJECT_SOURCE_DIR}/tests/convex_tests/*.cpp")
file(GLOB unit_tests "${PROJECT_SOURCE_DIR}/tests/unit_tests/*.cpp")

foreach(file ${convex_tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    "convex_tests/${name}.cpp"
    "${PROJECT_SOURCE_DIR}/tests/main.cpp"
    "${PROJECT_SOURCE_DIR}/tests/test_utils.cpp")
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