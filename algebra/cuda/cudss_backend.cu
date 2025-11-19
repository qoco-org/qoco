/**
 * @file cudss_backend.cu
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "cudss_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "kkt.h"
#include "common_linalg.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while(0)

#define CUDSS_CHECK(call) \
  do { \
    cudssStatus_t err = call; \
    if (err != CUDSS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDSS error at %s:%d\n", __FILE__, __LINE__); \
      exit(1); \
    } \
  } while(0)

// Contains data for linear system.
struct LinSysData {
  /** KKT matrix in CSC form (host). */
  QOCOCscMatrix* K;

  /** KKT matrix in CSR form (device) for cuDSS. */
  cudssMatrix_t K_csr;

  /** Permutation vector (not used for CUDA backend - kept for compatibility). */
  QOCOInt* p;

  /** Inverse of permutation vector (not used for CUDA backend - kept for compatibility). */
  QOCOInt* pinv;

  /** cuDSS handle. */
  cudssHandle_t handle;

  /** cuDSS config. */
  cudssConfig_t config;

  /** cuDSS data. */
  cudssData_t data;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff1;

  /** Buffer of size n + m + p (device) - used for b and x vectors. */
  QOCOFloat* d_xyzbuff1;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff2;

  /** Buffer of size n + m + p (device). */
  QOCOFloat* d_xyzbuff2;

  /** Device buffer for rhs (RHS of KKT system) - used during solve phase. */
  QOCOFloat* d_rhs;

  /** Device buffer for xyz (solution of KKT system) - used during solve phase. */
  QOCOFloat* d_xyz;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QOCOInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QOCOInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QOCOInt* PregtoKKT;

  /** Mapping from elements in At to elements in the KKT matrix. */
  QOCOInt* AttoKKT;

  /** Mapping from elements in Gt to elements in the KKT matrix. */
  QOCOInt* GttoKKT;

  QOCOInt Wnnz;

  /** cuSPARSE handle. */
  cusparseHandle_t cusparse_handle;

  /** Matrix description. */
  cusparseMatDescr_t descr;

  /** cuBLAS handle (for axpy operations). */
  cublasHandle_t cublas_handle;

  /** cuDSS dense matrix wrappers for solution and RHS vectors. */
  cudssMatrix_t d_rhs_matrix;
  cudssMatrix_t d_xyz_matrix;
};

// Convert CSC to CSR on host and copy to device
static void csc_to_csr_device(const QOCOCscMatrix* csc, QOCOInt** csr_row_ptr, 
                               QOCOInt** csr_col_ind, QOCOFloat** csr_val,
                               cusparseHandle_t handle)
{
  QOCOInt n = csc->n;
  QOCOInt m = csc->m;
  QOCOInt nnz = csc->nnz;
  
  // Allocate CSR arrays on host
  QOCOInt* h_csr_row_ptr = (QOCOInt*)qoco_calloc((m + 1), sizeof(QOCOInt));
  QOCOInt* h_csr_col_ind = (QOCOInt*)qoco_malloc(nnz * sizeof(QOCOInt));
  QOCOFloat* h_csr_val = (QOCOFloat*)qoco_malloc(nnz * sizeof(QOCOFloat));
  
  // Count nonzeros per row
  for (QOCOInt col = 0; col < n; ++col) {
    for (QOCOInt idx = csc->p[col]; idx < csc->p[col + 1]; ++idx) {
      QOCOInt row = csc->i[idx];
      h_csr_row_ptr[row + 1]++;
    }
  }
  
  // Compute row pointers (prefix sum)
  for (QOCOInt i = 0; i < m; ++i) {
    h_csr_row_ptr[i + 1] += h_csr_row_ptr[i];
  }
  
  // Temporary array to track current position in each row
  QOCOInt* row_pos = (QOCOInt*)qoco_malloc(m * sizeof(QOCOInt));
  for (QOCOInt i = 0; i < m; ++i) {
    row_pos[i] = h_csr_row_ptr[i];
  }
  
  // Fill CSR arrays
  for (QOCOInt col = 0; col < n; ++col) {
    for (QOCOInt idx = csc->p[col]; idx < csc->p[col + 1]; ++idx) {
      QOCOInt row = csc->i[idx];
      QOCOInt csr_idx = row_pos[row]++;
      h_csr_col_ind[csr_idx] = col;
      h_csr_val[csr_idx] = csc->x[idx];
    }
  }
  
  qoco_free(row_pos);
  
  // Allocate device memory
  CUDA_CHECK(cudaMalloc(csr_row_ptr, (m + 1) * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMalloc(csr_col_ind, nnz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMalloc(csr_val, nnz * sizeof(QOCOFloat)));
  
  // Copy to device
  CUDA_CHECK(cudaMemcpy(*csr_row_ptr, h_csr_row_ptr, (m + 1) * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_col_ind, h_csr_col_ind, nnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_val, h_csr_val, nnz * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  
  // Free host memory
  qoco_free(h_csr_row_ptr);
  qoco_free(h_csr_col_ind);
  qoco_free(h_csr_val);
}

static LinSysData* cudss_setup(QOCOProblemData* data, QOCOSettings* settings,
                               QOCOInt Wnnz)
{
  QOCOInt Kn = data->n + data->m + data->p;

  LinSysData* linsys_data = (LinSysData*)qoco_malloc(sizeof(LinSysData));

  // Initialize cuDSS
  #ifdef HAVE_CUDSS
  CUDSS_CHECK(cudssCreate(&linsys_data->handle));
  CUDSS_CHECK(cudssConfigCreate(&linsys_data->config));
  CUDSS_CHECK(cudssDataCreate(linsys_data->handle, &linsys_data->data));
  #else
  // cuDSS not available - set handles to NULL
  linsys_data->handle = NULL;
  linsys_data->config = NULL;
  linsys_data->data = NULL;
  #endif

  // Initialize cuSPARSE
  cusparseCreate(&linsys_data->cusparse_handle);
  cusparseCreateMatDescr(&linsys_data->descr);
  cusparseSetMatType(linsys_data->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(linsys_data->descr, CUSPARSE_INDEX_BASE_ZERO);

  // Initialize cuBLAS
  cublasCreate(&linsys_data->cublas_handle);

  // Allocate vector buffers
  linsys_data->xyzbuff1 = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * Kn);
  linsys_data->xyzbuff2 = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * Kn);
  CUDA_CHECK(cudaMalloc(&linsys_data->d_xyzbuff1, sizeof(QOCOFloat) * Kn));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_xyzbuff2, sizeof(QOCOFloat) * Kn));
  linsys_data->Wnnz = Wnnz;

  // Allocate memory for mappings to KKT matrix
  linsys_data->nt2kkt = (QOCOInt*)qoco_calloc(Wnnz, sizeof(QOCOInt));
  linsys_data->ntdiag2kkt = (QOCOInt*)qoco_calloc(data->m, sizeof(QOCOInt));
  linsys_data->PregtoKKT = data->P ? (QOCOInt*)qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  linsys_data->AttoKKT = (QOCOInt*)qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  linsys_data->GttoKKT = (QOCOInt*)qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  QOCOInt* nt2kkt_temp = (QOCOInt*)qoco_calloc(Wnnz, sizeof(QOCOInt));
  QOCOInt* ntdiag2kkt_temp = (QOCOInt*)qoco_calloc(data->m, sizeof(QOCOInt));
  QOCOInt* PregtoKKT_temp = data->P ? (QOCOInt*)qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  QOCOInt* AttoKKT_temp = (QOCOInt*)qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  QOCOInt* GttoKKT_temp = (QOCOInt*)qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  // Construct KKT matrix (no permutation for CUDA backend)
  linsys_data->K = construct_kkt(
      data->P ? get_csc_matrix(data->P) : NULL, get_csc_matrix(data->A), get_csc_matrix(data->G),
      get_csc_matrix(data->At), get_csc_matrix(data->Gt),
      settings->kkt_static_reg, data->n, data->m, data->p, data->l, data->nsoc,
      data->q, PregtoKKT_temp, AttoKKT_temp, GttoKKT_temp, nt2kkt_temp,
      ntdiag2kkt_temp, Wnnz);

  // No AMD ordering or permutation - use KKT matrix directly
  // Set identity permutation (no-op)
  linsys_data->p = (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  linsys_data->pinv = (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  for (QOCOInt i = 0; i < linsys_data->K->n; ++i) {
    linsys_data->p[i] = i;
    linsys_data->pinv[i] = i;
  }

  // Copy mappings directly (no permutation)
  for (QOCOInt i = 0; i < Wnnz; ++i) {
    linsys_data->nt2kkt[i] = nt2kkt_temp[i];
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    linsys_data->ntdiag2kkt[i] = ntdiag2kkt_temp[i];
  }

  if (data->P && PregtoKKT_temp) {
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i) {
      linsys_data->PregtoKKT[i] = PregtoKKT_temp[i];
    }
  }

  for (QOCOInt i = 0; i < get_nnz(data->A); ++i) {
    linsys_data->AttoKKT[i] = AttoKKT_temp[i];
  }

  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->GttoKKT[i] = GttoKKT_temp[i];
  }

  qoco_free(nt2kkt_temp);
  qoco_free(ntdiag2kkt_temp);
  qoco_free(PregtoKKT_temp);
  qoco_free(AttoKKT_temp);
  qoco_free(GttoKKT_temp);

  // Convert KKT matrix from CSC (CPU) to CSR (GPU) for cuDSS
  // KKT matrix is formed on CPU in CSC, now convert and move to GPU
  // Use csc_to_csr_device helper function
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;
  
  csc_to_csr_device(linsys_data->K, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle);

  // Create cuDSS matrix in CSR format
  #ifdef HAVE_CUDSS
  // cuDSS CSR format needs rowStart and rowEnd arrays
  // For standard CSR, rowEnd[i] = rowStart[i+1], so we can create it
  QOCOInt* csr_row_end;
  CUDA_CHECK(cudaMalloc(&csr_row_end, Kn * sizeof(QOCOInt)));
  // Copy rowStart[1..n] to rowEnd[0..n-1]
  CUDA_CHECK(cudaMemcpy(csr_row_end, &csr_row_ptr[1], Kn * sizeof(QOCOInt), cudaMemcpyDeviceToDevice));
  
  // Determine data types
  #include <library_types.h>
  cudaDataType_t indexType = CUDA_R_32F;  // Use 32-bit float as placeholder for 32-bit int indices
  cudaDataType_t valueType_setup = (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;
  
  CUDSS_CHECK(cudssMatrixCreateCsr(&linsys_data->K_csr,
                                   (int64_t)Kn, (int64_t)Kn, (int64_t)linsys_data->K->nnz,
                                   csr_row_ptr, csr_row_end, csr_col_ind, csr_val,
                                   indexType, valueType_setup,
                                   CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
  
  // Note: We'll create dense matrix wrappers after allocating the device buffers
  // (they're created later after d_rhs_ptr and d_xyz_ptr are allocated)
  #else
  // cuDSS not available - set to NULL (solve will use fallback)
  linsys_data->K_csr = NULL;
  linsys_data->d_rhs_matrix = NULL;
  linsys_data->d_xyz_matrix = NULL;
  // Free device arrays since cuDSS won't manage them
  cudaFree(csr_row_ptr);
  cudaFree(csr_col_ind);
  cudaFree(csr_val);
  #endif

  // Allocate device vectors for rhs and xyz
  // These will be used by the cuDSS dense matrix wrappers
  QOCOFloat* d_rhs_ptr = NULL;
  QOCOFloat* d_xyz_ptr = NULL;
  CUDA_CHECK(cudaMalloc(&d_rhs_ptr, Kn * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMalloc(&d_xyz_ptr, Kn * sizeof(QOCOFloat)));
  
  // Store in struct for cleanup
  linsys_data->d_xyzbuff1 = d_rhs_ptr;  // Used for rhs buffer
  linsys_data->d_xyzbuff2 = d_xyz_ptr;  // Used for xyz buffer
  
  // Create dense matrix wrappers for solution and RHS vectors (column vectors)
  // Note: d_rhs_matrix wraps d_xyzbuff1, d_xyz_matrix wraps d_xyzbuff2
  #ifdef HAVE_CUDSS
  // Use valueType_setup declared above
  CUDSS_CHECK(cudssMatrixCreateDn(&linsys_data->d_rhs_matrix, (int64_t)Kn, 1, (int64_t)Kn, linsys_data->d_xyzbuff1, valueType_setup, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&linsys_data->d_xyz_matrix, (int64_t)Kn, 1, (int64_t)Kn, linsys_data->d_xyzbuff2, valueType_setup, CUDSS_LAYOUT_COL_MAJOR));
  #else
  linsys_data->d_rhs_matrix = NULL;
  linsys_data->d_xyz_matrix = NULL;
  #endif

  return linsys_data;
}

static void cudss_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  // Update matrix values - KKT matrix was updated on host (in CSC format)
  // Convert updated CSC to CSR and update device matrix
  // Destroy old matrix if it exists
  #ifdef HAVE_CUDSS
  if (linsys_data->K_csr) {
    cudssMatrixDestroy(linsys_data->K_csr);
    linsys_data->K_csr = NULL;
  }
  #endif
  
  // Free old CSR device arrays if they exist (they should be managed by cuDSS, but be safe)
  // Note: cuDSS manages the CSR arrays, so we don't free them here
  
  // Reconvert KKT matrix from CSC (host) to CSR (device)
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;
  
  csc_to_csr_device(linsys_data->K, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle);
  
  // Create new cuDSS matrix
  #ifdef HAVE_CUDSS
  // cuDSS CSR format needs rowStart and rowEnd arrays
  QOCOInt* csr_row_end;
  CUDA_CHECK(cudaMalloc(&csr_row_end, linsys_data->K->n * sizeof(QOCOInt)));
  // Copy rowStart[1..n] to rowEnd[0..n-1]
  CUDA_CHECK(cudaMemcpy(csr_row_end, &csr_row_ptr[1], linsys_data->K->n * sizeof(QOCOInt), cudaMemcpyDeviceToDevice));
  
  // Determine data types
  #include <library_types.h>
  cudaDataType_t indexType_factor = CUDA_R_32F;  // Use 32-bit float as placeholder for 32-bit int indices
  cudaDataType_t valueType_factor = (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;
  
  CUDSS_CHECK(cudssMatrixCreateCsr(&linsys_data->K_csr,
                                   (int64_t)linsys_data->K->n, (int64_t)linsys_data->K->n, (int64_t)linsys_data->K->nnz,
                                   csr_row_ptr, csr_row_end, csr_col_ind, csr_val,
                                   indexType_factor, valueType_factor,
                                   CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
  
  // Perform analysis and factorization - use dense matrix wrappers for dummy vectors
  CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_ANALYSIS,
                           linsys_data->config, linsys_data->data,
                           linsys_data->K_csr, linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix));

  CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_FACTORIZATION,
                           linsys_data->config, linsys_data->data,
                           linsys_data->K_csr, linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix));
  #else
  // cuDSS not available - free device arrays and set to NULL
  linsys_data->K_csr = NULL;
  cudaFree(csr_row_ptr);
  cudaFree(csr_col_ind);
  cudaFree(csr_val);
  // cuDSS not available - skip analysis/factorization (solve will use fallback)
  #endif
}

// Helper function to map work->rhs and work->xyz to device buffers during solve phase
extern "C" {
void map_work_buffers_to_device(void* linsys_data_ptr, QOCOWorkspace* work)
{
  // During solve phase, functions constructing rhs/xyz will operate on device buffers
  // This function is called at start of solve phase
  // Actual mapping is handled by having functions use device buffers from LinSysData
  (void)linsys_data_ptr;
  (void)work;
}

QOCOFloat* get_device_rhs(void* linsys_data_ptr)
{
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  return linsys_data->d_rhs;
}

QOCOFloat* get_device_xyz(void* linsys_data_ptr)
{
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  return linsys_data->d_xyz;
}

void unmap_work_buffers_from_device(void* linsys_data_ptr, QOCOWorkspace* work)
{
  // Copy final results from device buffers back to host buffers
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  QOCOInt n = work->data->n + work->data->m + work->data->p;
  
  // Copy d_xyz back to work->xyz (d_xyz contains final solution from last solve)
  CUDA_CHECK(cudaMemcpy(work->xyz, linsys_data->d_xyz, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  // d_rhs doesn't need to be copied back as it's only used internally
}
}

static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOFloat* b, QOCOFloat* x, QOCOInt iter_ref_iters)
{
  QOCOInt n = linsys_data->K->n;
  (void)iter_ref_iters;  // No iterative refinement for CUDA backend

  // During solve phase, b and x should already be device pointers
  // (get_data_vectorf returns device pointers during solve phase)
  // Use them directly - no copying needed
  
  // Check if pointers are valid device pointers (always check, even in fallback)
  cudaPointerAttributes attrs_b, attrs_x;
  cudaError_t err_b = cudaPointerGetAttributes(&attrs_b, b);
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  
  // If b or x are not device pointers, we need to copy from host to device first
  QOCOFloat* d_b = b;
  QOCOFloat* d_x = x;
  
  if (err_b != cudaSuccess || attrs_b.type != cudaMemoryTypeDevice) {
    // b is on host - copy to device
    fprintf(stderr, "WARNING: b is on host, copying to device (err=%d, type=%d)\n", 
            err_b, (err_b == cudaSuccess) ? attrs_b.type : -1);
    d_b = linsys_data->d_xyzbuff1;  // Use temporary buffer
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  }
  
  if (err_x != cudaSuccess || attrs_x.type != cudaMemoryTypeDevice) {
    // x is on host - use device buffer
    fprintf(stderr, "WARNING: x is on host, using device buffer (err=%d, type=%d)\n", 
            err_x, (err_x == cudaSuccess) ? attrs_x.type : -1);
    d_x = linsys_data->d_xyzbuff2;  // Use temporary buffer
  }
  
  // Solve - cuDSS expects dense matrix wrappers for b and x
  // Copy data from b and x into cuDSS dense matrix buffers
  CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff1, d_b, n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
  
  // cuDSS API signature: cudssExecute(handle, phase, config, data, matrix, solution, rhs)
  // where solution is the output and rhs is the input
  #ifdef HAVE_CUDSS
  fprintf(stderr, "DEBUG: Calling cuDSS solve with HAVE_CUDSS defined\n");
  cudssStatus_t status = cudssExecute(linsys_data->handle, CUDSS_PHASE_SOLVE,
                                      linsys_data->config, linsys_data->data,
                                      linsys_data->K_csr, linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    fprintf(stderr, "ERROR: cuDSS solve failed with status %d\n", (int)status);
  }
  // Copy solution back from cuDSS dense matrix buffer to x
  CUDA_CHECK(cudaMemcpy(d_x, linsys_data->d_xyzbuff2, n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
  #else
  fprintf(stderr, "DEBUG: HAVE_CUDSS not defined, using fallback (copy b to x)\n");
  // cuDSS not available - use fallback: copy solution from RHS (will fail convergence but won't crash)
  // TODO: Implement proper solve using cuSPARSE/cuSOLVER when cuDSS is not available
  CUDA_CHECK(cudaMemcpy(d_x, d_b, n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
  #endif

  // DEBUG: Copy result back to CPU and print
  QOCOFloat* x_host = (QOCOFloat*)malloc(n * sizeof(QOCOFloat));
  QOCOFloat* b_host = (QOCOFloat*)malloc(n * sizeof(QOCOFloat));
  CUDA_CHECK(cudaMemcpy(x_host, d_x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b_host, d_b, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  
  fprintf(stderr, "\n=== DEBUG: After cudss_solve (n=%d) ===\n", (int)n);
  fprintf(stderr, "RHS (b): ");
  for (QOCOInt i = 0; i < n && i < 20; ++i) {
    fprintf(stderr, "%e ", b_host[i]);
  }
  if (n > 20) fprintf(stderr, "...");
  fprintf(stderr, "\n");
  
  fprintf(stderr, "Solution (x): ");
  for (QOCOInt i = 0; i < n && i < 20; ++i) {
    fprintf(stderr, "%e ", x_host[i]);
  }
  if (n > 20) fprintf(stderr, "...");
  fprintf(stderr, "\n");
  fprintf(stderr, "=====================================\n\n");
  
  free(x_host);
  free(b_host);

  // Copy result back to original pointer if it was on host
  if (err_x != cudaSuccess || attrs_x.type != cudaMemoryTypeDevice) {
    // x was on host - copy result back
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  }
  
  // During solve phase, don't copy back to host for device pointers
  // Result stays on device until solver terminates
}

static void cudss_initialize_nt(LinSysData* linsys_data, QOCOInt m)
{
  for (QOCOInt i = 0; i < linsys_data->Wnnz; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[i]] = 0.0;
  }

  // Set Nesterov-Todd block in KKT matrix to -I
  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] = -1.0;
  }

  // Update device matrix - redo CSR conversion
  // In production, we'd update device matrix directly
  // For now, we'll need to redo the conversion in factor function
}

static void cudss_update_nt(LinSysData* linsys_data, QOCOFloat* WtW,
                            QOCOFloat kkt_static_reg, QOCOInt m)
{
  for (QOCOInt i = 0; i < linsys_data->Wnnz; ++i) {
    linsys_data->K->x[linsys_data->nt2kkt[i]] = -WtW[i];
  }

  // Regularize Nesterov-Todd block of KKT matrix
  for (QOCOInt i = 0; i < m; ++i) {
    linsys_data->K->x[linsys_data->ntdiag2kkt[i]] -= kkt_static_reg;
  }
}

static void cudss_update_data(LinSysData* linsys_data, QOCOProblemData* data)
{
  // Update P in KKT matrix
  if (data->P && linsys_data->PregtoKKT) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i) {
      linsys_data->K->x[linsys_data->PregtoKKT[i]] = Pcsc->x[i];
    }
  }

  // Update A in KKT matrix
  QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
  for (QOCOInt i = 0; i < get_nnz(data->A); ++i) {
    linsys_data->K->x[linsys_data->AttoKKT[data->AtoAt[i]]] = Acsc->x[i];
  }

  // Update G in KKT matrix
  QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->K->x[linsys_data->GttoKKT[data->GtoGt[i]]] = Gcsc->x[i];
  }

  // Update device matrix - convert to CSR and update
  // Note: This is simplified; in production we'd maintain device copy
}

static void cudss_cleanup(LinSysData* linsys_data)
{
  if (linsys_data->K_csr) {
    cudssMatrixDestroy(linsys_data->K_csr);
  }
  if (linsys_data->d_rhs_matrix) {
    cudssMatrixDestroy(linsys_data->d_rhs_matrix);
  }
  if (linsys_data->d_xyz_matrix) {
    cudssMatrixDestroy(linsys_data->d_xyz_matrix);
  }
  if (linsys_data->data) {
    cudssDataDestroy(linsys_data->handle, linsys_data->data);
  }
  if (linsys_data->config) {
    cudssConfigDestroy(linsys_data->config);
  }
  if (linsys_data->handle) {
    cudssDestroy(linsys_data->handle);
  }
  if (linsys_data->cusparse_handle) {
    cusparseDestroy(linsys_data->cusparse_handle);
  }
  if (linsys_data->descr) {
    cusparseDestroyMatDescr(linsys_data->descr);
  }
  if (linsys_data->cublas_handle) {
    cublasDestroy(linsys_data->cublas_handle);
  }
  free_qoco_csc_matrix(linsys_data->K);
  qoco_free(linsys_data->p);
  qoco_free(linsys_data->pinv);
  qoco_free(linsys_data->xyzbuff1);
  qoco_free(linsys_data->xyzbuff2);
  if (linsys_data->d_xyzbuff1) cudaFree(linsys_data->d_xyzbuff1);
  if (linsys_data->d_xyzbuff2) cudaFree(linsys_data->d_xyzbuff2);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  qoco_free(linsys_data);
}

// Export the backend struct
LinSysBackend backend = {.linsys_setup = cudss_setup,
                         .linsys_initialize_nt = cudss_initialize_nt,
                         .linsys_update_nt = cudss_update_nt,
                         .linsys_update_data = cudss_update_data,
                         .linsys_factor = cudss_factor,
                         .linsys_solve = cudss_solve,
                         .linsys_cleanup = cudss_cleanup};

