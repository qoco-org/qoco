cmake_minimum_required(VERSION 3.18)

set(
	amd_src
	amd_1.c
    amd_2.c
    amd_aat.c
    amd_control.c
    amd_defaults.c
    amd_dump.c
    amd_info.c
    amd_order.c
    amd_post_tree.c
    amd_postorder.c
    amd_preprocess.c
    amd_valid.c
    amd_version.c
    SuiteSparse_config.c
	)

set(
	amd_headers
    amd.h
	amd_internal.h
    SuiteSparse_config.h
	)

# Disable timer to build on Windows.
add_compile_definitions(NTIMER)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library (amd OBJECT ${amd_src} ${amd_headers})
target_include_directories(amd PRIVATE ${PROJECT_SOURCE_DIR}/include)
