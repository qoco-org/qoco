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

  /** Permutation vector. */
  QOCOInt* p;

  /** Inverse of permutation vector. */
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
  CUDSS_CHECK(cudssCreate(&linsys_data->handle));
  CUDSS_CHECK(cudssConfigCreate(&linsys_data->config));
  CUDSS_CHECK(cudssDataCreate(linsys_data->handle, &linsys_data->data));

  // Initialize cuSPARSE
  cusparseCreate(&linsys_data->cusparse_handle);
  cusparseCreateMatDescr(&linsys_data->descr);
  cusparseSetMatType(linsys_data->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(linsys_data->descr, CUSPARSE_INDEX_BASE_ZERO);

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

  // Construct KKT matrix
  linsys_data->K = construct_kkt(
      data->P ? get_csc_matrix(data->P) : NULL, get_csc_matrix(data->A), get_csc_matrix(data->G),
      get_csc_matrix(data->At), get_csc_matrix(data->Gt),
      settings->kkt_static_reg, data->n, data->m, data->p, data->l, data->nsoc,
      data->q, PregtoKKT_temp, AttoKKT_temp, GttoKKT_temp, nt2kkt_temp,
      ntdiag2kkt_temp, Wnnz);

  // Compute AMD ordering
  linsys_data->p = (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  linsys_data->pinv = (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  QOCOInt amd_status =
      amd_order(linsys_data->K->n, linsys_data->K->p, linsys_data->K->i,
                linsys_data->p, (double*)NULL, (double*)NULL);
  if (amd_status < 0) {
    return NULL;
  }
  invert_permutation(linsys_data->p, linsys_data->pinv, linsys_data->K->n);

  // Permute KKT matrix
  QOCOInt* KtoPKPt = (QOCOInt*)qoco_malloc(linsys_data->K->nnz * sizeof(QOCOInt));
  QOCOCscMatrix* PKPt = csc_symperm(linsys_data->K, linsys_data->pinv, KtoPKPt);

  // Update mappings to permuted matrix
  for (QOCOInt i = 0; i < Wnnz; ++i) {
    linsys_data->nt2kkt[i] = KtoPKPt[nt2kkt_temp[i]];
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    linsys_data->ntdiag2kkt[i] = KtoPKPt[ntdiag2kkt_temp[i]];
  }

  if (data->P && PregtoKKT_temp) {
    for (QOCOInt i = 0; i < get_nnz(data->P); ++i) {
      linsys_data->PregtoKKT[i] = KtoPKPt[PregtoKKT_temp[i]];
    }
  }

  for (QOCOInt i = 0; i < get_nnz(data->A); ++i) {
    linsys_data->AttoKKT[i] = KtoPKPt[AttoKKT_temp[i]];
  }

  for (QOCOInt i = 0; i < get_nnz(data->G); ++i) {
    linsys_data->GttoKKT[i] = KtoPKPt[GttoKKT_temp[i]];
  }

  free_qoco_csc_matrix(linsys_data->K);
  qoco_free(KtoPKPt);
  qoco_free(nt2kkt_temp);
  qoco_free(ntdiag2kkt_temp);
  qoco_free(PregtoKKT_temp);
  qoco_free(AttoKKT_temp);
  qoco_free(GttoKKT_temp);
  linsys_data->K = PKPt;

  // Convert KKT matrix from CSC (CPU) to CSR (GPU) for cuDSS
  // KKT matrix is formed on CPU in CSC, now convert and move to GPU
  // Use csc_to_csr_device helper function
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;
  
  csc_to_csr_device(linsys_data->K, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle);

  // Create cuDSS matrix in CSR format
  CUDSS_CHECK(cudssMatrixCreateCsr(linsys_data->handle, &linsys_data->K_csr,
                                   linsys_data->K->n, linsys_data->K->n,
                                   linsys_data->K->nnz, csr_row_ptr, csr_col_ind, csr_val));

  // Allocate device vectors
  QOCOFloat* d_b = NULL;
  QOCOFloat* d_x = NULL;
  CUDA_CHECK(cudaMalloc(&d_b, Kn * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMalloc(&d_x, Kn * sizeof(QOCOFloat)));
  
  // Store in struct for cleanup
  linsys_data->d_xyzbuff1 = d_b;  // Reuse for b
  linsys_data->d_xyzbuff2 = d_x;  // Reuse for x
  
  // For cuDSS, vectors are passed directly as device pointers
  // We'll handle this in solve function

  return linsys_data;
}

static void cudss_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  // Update matrix values - KKT matrix was updated on host (in CSC format)
  // Convert updated CSC to CSR and update device matrix
  // Destroy old matrix if it exists
  if (linsys_data->K_csr) {
    cudssMatrixDestroy(linsys_data->K_csr);
    linsys_data->K_csr = NULL;
  }
  
  // Free old CSR device arrays if they exist (they should be managed by cuDSS, but be safe)
  // Note: cuDSS manages the CSR arrays, so we don't free them here
  
  // Reconvert KKT matrix from CSC (host) to CSR (device)
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;
  
  csc_to_csr_device(linsys_data->K, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle);
  
  // Create new cuDSS matrix
  CUDSS_CHECK(cudssMatrixCreateCsr(linsys_data->handle, &linsys_data->K_csr,
                                   linsys_data->K->n, linsys_data->K->n,
                                   linsys_data->K->nnz, csr_row_ptr, csr_col_ind, csr_val));
  
  // Perform analysis and factorization
  QOCOFloat* d_dummy = linsys_data->d_xyzbuff1;  // Dummy vector for API
  (void)d_dummy;  // Suppress unused variable warning
  CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_ANALYSIS,
                           linsys_data->config, linsys_data->data,
                           linsys_data->K_csr, linsys_data->d_xyzbuff1, linsys_data->d_xyzbuff1));

  CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_FACTORIZATION,
                           linsys_data->config, linsys_data->data,
                           linsys_data->K_csr, linsys_data->d_xyzbuff1, linsys_data->d_xyzbuff1));
}

static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOFloat* b, QOCOFloat* x, QOCOInt iter_ref_iters)
{
  QOCOInt n = linsys_data->K->n;

  // b and x are host pointers, but vectors in workspace are on device
  // Use workspace buffers directly on device
  QOCOFloat* d_b_ptr = linsys_data->d_xyzbuff1;  // b vector
  QOCOFloat* d_x_ptr = linsys_data->d_xyzbuff2;  // x vector
  
  // Permute b and copy to device buffer (only once, at start)
  for (QOCOInt i = 0; i < n; ++i) {
    linsys_data->xyzbuff1[i] = b[linsys_data->p[i]];
  }
  CUDA_CHECK(cudaMemcpy(d_b_ptr, linsys_data->xyzbuff1,
                        n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));

  // Solve - cuDSS expects device pointers for b and x
  CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_SOLVE,
                           linsys_data->config, linsys_data->data,
                           linsys_data->K_csr, d_x_ptr, d_b_ptr));

  // Iterative refinement
  // d_b_ptr holds permuted b (rhs)
  // d_x_ptr holds solution x
  for (QOCOInt i = 0; i < iter_ref_iters; ++i) {
    // Save current x to host buffer before compute residual
    CUDA_CHECK(cudaMemcpy(linsys_data->xyzbuff1, d_x_ptr,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
    for (QOCOInt k = 0; k < n; ++k) {
      x[linsys_data->p[k]] = linsys_data->xyzbuff1[k];
    }

    // Compute residual: r = b - K * x
    // Use cuSPARSE for sparse matrix-vector products on device
    // For now, compute on host using existing functions but with proper linkage
    // TODO: Implement full CUDA version of kkt_multiply
    // Since kkt_multiply uses many C functions (USpMv_matrix, SpMtv_matrix, nt_multiply),
    // we'll keep it as a host function call for now, but ensure data is on host
    kkt_multiply(x, linsys_data->xyzbuff2, work->data, work->Wfull, work->xbuff,
                 work->ubuff1, work->ubuff2);
    for (QOCOInt k = 0; k < n; ++k) {
      x[k] = linsys_data->xyzbuff2[linsys_data->p[k]];
    }

    for (QOCOInt j = 0; j < n; ++j) {
      x[j] = b[j] - x[j];  // x now holds residual r
    }

    // Copy residual r to device (permuted) in d_b_ptr
    for (QOCOInt k = 0; k < n; ++k) {
      linsys_data->xyzbuff1[k] = x[linsys_data->p[k]];
    }
    CUDA_CHECK(cudaMemcpy(d_b_ptr, linsys_data->xyzbuff1,
                          n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));

    // Save current solution x before solve overwrites d_x_ptr
    CUDA_CHECK(cudaMemcpy(linsys_data->xyzbuff1, d_x_ptr,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));

    // dx = K \ r (solve for correction, overwrites d_x_ptr with dx)
    CUDSS_CHECK(cudssExecute(linsys_data->handle, CUDSS_PHASE_SOLVE,
                             linsys_data->config, linsys_data->data,
                             linsys_data->K_csr, d_x_ptr, d_b_ptr));

    // x_new = x_old + dx (accumulate correction on device)
    const QOCOFloat one = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // d_x_ptr has dx, linsys_data->xyzbuff1 has x_old (on host)
    // Copy x_old to d_b_ptr (temporary), then accumulate: d_x_ptr = d_b_ptr + d_x_ptr
    CUDA_CHECK(cudaMemcpy(d_b_ptr, linsys_data->xyzbuff1,
                          n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
    cublasDaxpy(handle, n, &one, d_x_ptr, 1, d_b_ptr, 1);  // d_b_ptr = d_b_ptr + d_x_ptr
    // Swap so d_x_ptr holds updated solution
    QOCOFloat* temp = d_x_ptr;
    d_x_ptr = d_b_ptr;
    d_b_ptr = temp;
    cublasDestroy(handle);
  }

  // Copy final result back to host
  CUDA_CHECK(cudaMemcpy(linsys_data->xyzbuff1, d_x_ptr,
                        n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  
  for (QOCOInt i = 0; i < n; ++i) {
    x[linsys_data->p[i]] = linsys_data->xyzbuff1[i];
  }
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
  if (linsys_data->data) {
    cudssDataDestroy(linsys_data->data);
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

