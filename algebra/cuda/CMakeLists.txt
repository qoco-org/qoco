# Set CUDA flags before enabling language to allow unsupported GCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set environment variable as fallback for compiler ID test
set(ENV{CMAKE_CUDA_FLAGS} "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_library(CUBLAS_LIB cublas PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUSPARSE_LIB cusparse PATHS ${CUDAToolkit_LIBRARY_DIR})

# Find cuDSS - may need adjustment based on installation
find_library(CUDSS_LIB cudss 
  PATHS 
    ${CUDAToolkit_LIBRARY_DIR}
    /usr/local/cuda/lib64
    /usr/local/cuda/lib
    ENV CUDA_PATH
    ENV CUDA_HOME
  PATH_SUFFIXES lib64 lib
)

if(NOT CUDSS_LIB)
  message(WARNING "cuDSS library not found. You may need to install cuDSS separately.")
  message(STATUS "cuDSS can be obtained from NVIDIA Developer website")
  message(STATUS "Building with cuDSS stubs - full functionality requires cuDSS installation")
  # Create a dummy library variable to avoid errors
  set(CUDSS_LIB "")
else()
  add_compile_definitions(HAVE_CUDSS)
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

