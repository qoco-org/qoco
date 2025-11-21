set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_library(CUBLAS_LIB cublas PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUSPARSE_LIB cusparse PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUDSS_LIB cudss PATHS ${CUDAToolkit_LIBRARY_DIR})

if(NOT CUBLAS_LIB)
  message(FATAL_ERROR "cuBLAS library not found.")
endif()
if(NOT CUSPARSE_LIB)
  message(FATAL_ERROR "cuSPARSE library not found.")
endif()
if(NOT CUDSS_LIB)
  message(FATAL_ERROR "cuDSS library not found.")
endif()

set_property(GLOBAL APPEND PROPERTY QOCO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.cu ${CMAKE_CURRENT_SOURCE_DIR}/cuda_linalg.cu)
set_property(GLOBAL APPEND PROPERTY QOCO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.h)

# Add CUDA libraries to global property (will be used in main CMakeLists.txt)
set_property(GLOBAL APPEND PROPERTY QOCO_CUDA_LIBS 
  ${CUBLAS_LIB} 
  ${CUSPARSE_LIB}
  ${CUDSS_LIB}
)