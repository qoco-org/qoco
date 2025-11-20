set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_library(CUBLAS_LIB cublas PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUSPARSE_LIB cusparse PATHS ${CUDAToolkit_LIBRARY_DIR})
# Search for cuDSS in CU_DSS_ROOT first, then system paths
if(DEFINED ENV{CU_DSS_ROOT})
  find_library(CUDSS_LIB cudss PATHS "$ENV{CU_DSS_ROOT}/lib" NO_DEFAULT_PATH)
endif()
if(NOT CUDSS_LIB)
  find_library(CUDSS_LIB cudss PATHS ${CUDAToolkit_LIBRARY_DIR})
endif()

if(NOT CUDSS_LIB)
  message(FATAL_ERROR "cuDSS library not found. You may need to install cuDSS separately.")
else()
  # Add include directory for cuDSS headers
  if(DEFINED ENV{CU_DSS_ROOT})
    include_directories("$ENV{CU_DSS_ROOT}/include")
  endif()
  message(STATUS "Found cuDSS library: ${CUDSS_LIB}")
endif()

set_property(GLOBAL APPEND PROPERTY QOCO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.cu ${CMAKE_CURRENT_SOURCE_DIR}/cuda_linalg.cu)
set_property(GLOBAL APPEND PROPERTY QOCO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.h)

# Add CUDA libraries to global property (will be used in main CMakeLists.txt)
set_property(GLOBAL APPEND PROPERTY QOCO_CUDA_LIBS 
  ${CUBLAS_LIB} 
  ${CUSPARSE_LIB}
  ${CUDSS_LIB}
)