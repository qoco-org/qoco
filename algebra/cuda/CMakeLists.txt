# Set CUDA flags before enabling language to allow unsupported GCC
# Set CMAKE_CUDA_FLAGS_INIT before enable_language so it's used during compiler ID test
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set environment variable as fallback for compiler ID test
set(ENV{CMAKE_CUDA_FLAGS} "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

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
  message(WARNING "cuDSS library not found. You may need to install cuDSS separately.")
  message(STATUS "cuDSS can be obtained from NVIDIA Developer website")
  message(STATUS "Building with cuDSS stubs - full functionality requires cuDSS installation")
  # Create a dummy library variable to avoid errors
  set(CUDSS_LIB "")
else()
  # Add HAVE_CUDSS definition - use CMAKE_CUDA_FLAGS to ensure it's passed to CUDA compiler
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHAVE_CUDSS")
  add_compile_definitions(HAVE_CUDSS)
  # Add include directory for cuDSS headers
  if(DEFINED ENV{CU_DSS_ROOT})
    include_directories("$ENV{CU_DSS_ROOT}/include")
  endif()
  message(STATUS "Found cuDSS library: ${CUDSS_LIB}")
  message(STATUS "Added HAVE_CUDSS definition to CUDA flags")
endif()

set_property(GLOBAL APPEND PROPERTY QOCO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.cu ${CMAKE_CURRENT_SOURCE_DIR}/cuda_linalg.cu)
set_property(GLOBAL APPEND PROPERTY QOCO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cudss_backend.h)

# Add CUDA libraries to global property (will be used in main CMakeLists.txt)
set_property(GLOBAL APPEND PROPERTY QOCO_CUDA_LIBS 
  ${CUBLAS_LIB} 
  ${CUSPARSE_LIB}
)

if(CUDSS_LIB)
  set_property(GLOBAL APPEND PROPERTY QOCO_CUDA_LIBS ${CUDSS_LIB})
endif()

