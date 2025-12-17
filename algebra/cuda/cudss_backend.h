/**
 * @file cudss_backend.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines the cuDSS backend.
 */

#ifndef CUDSS_BACKEND_H
#define CUDSS_BACKEND_H

#include "cuda_types.h"
#include <cudss.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <library_types.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "common_linalg.h"
#include "kkt.h"
#ifdef __cplusplus
}
#endif
#include "structs.h"

// Structure to hold function pointers for CUDA libraries
typedef struct {
  // cuDSS function pointers
  cudssStatus_t (*cudssCreate)(cudssHandle_t* handle);
  cudssStatus_t (*cudssConfigCreate)(cudssConfig_t* config);
  cudssStatus_t (*cudssDataCreate)(cudssHandle_t handle, cudssData_t* data);
  cudssStatus_t (*cudssMatrixCreateCsr)(cudssMatrix_t* mat, int64_t rows, int64_t cols, 
                                         int64_t nnz, void* csrRowPtr, void* csrRowPtr_type,
                                         void* csrColInd, void* csrVal,
                                         cudaDataType_t idxType, cudaDataType_t valType,
                                         cudssMatrixType_t type, int view,
                                         cudssIndexBase_t base);
  cudssStatus_t (*cudssExecute)(cudssHandle_t handle, cudssPhase_t phase,
                                 cudssConfig_t config, cudssData_t data,
                                 cudssMatrix_t matA, cudssMatrix_t matB, cudssMatrix_t matC);
  cudssStatus_t (*cudssMatrixCreateDn)(cudssMatrix_t* mat, int64_t rows, int64_t cols,
                                        int64_t ld, void* data, cudaDataType_t type,
                                        int layout);
  cudssStatus_t (*cudssMatrixSetValues)(cudssMatrix_t mat, void* data);
  cudssStatus_t (*cudssMatrixDestroy)(cudssMatrix_t mat);
  cudssStatus_t (*cudssDataDestroy)(cudssHandle_t handle, cudssData_t data);
  cudssStatus_t (*cudssConfigDestroy)(cudssConfig_t config);
  cudssStatus_t (*cudssDestroy)(cudssHandle_t handle);
  
  // cuSPARSE function pointers
  cusparseStatus_t (*cusparseCreate)(cusparseHandle_t* handle);
  cusparseStatus_t (*cusparseCreateMatDescr)(cusparseMatDescr_t* descr);
  cusparseStatus_t (*cusparseSetMatType)(cusparseMatDescr_t descr, cusparseMatrixType_t type);
  cusparseStatus_t (*cusparseSetMatIndexBase)(cusparseMatDescr_t descr, cusparseIndexBase_t base);
  cusparseStatus_t (*cusparseDestroy)(cusparseHandle_t handle);
  cusparseStatus_t (*cusparseDestroyMatDescr)(cusparseMatDescr_t descr);
  
  // cuBLAS function pointers
  cublasStatus_t (*cublasCreate)(cublasHandle_t* handle);
  cublasStatus_t (*cublasDdot)(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  cublasStatus_t (*cublasDestroy)(cublasHandle_t handle);
  cublasStatus_t (*cublasIdamin)(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  cublasStatus_t (*cublasIdamax)(cublasHandle_t handle, int n, const double* x, int incx, int* result);
} CudaLibFuncs;

// Accessor function for CUDA library function pointers
// Note: Call load_cuda_libraries() before using these pointers
CudaLibFuncs* get_cuda_funcs(void);

// Load CUDA libraries dynamically (must be called before using CUDA library functions)
int load_cuda_libraries(void);

extern LinSysBackend backend;

#endif /* #ifndef CUDSS_BACKEND_H */
