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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <library_types.h>

#include "common_linalg.h"
#include "kkt.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t err = call;                                                  \
    if (err != CUDSS_STATUS_SUCCESS) {                                         \
      const char* err_str =                                                    \
          (err == CUDSS_STATUS_NOT_INITIALIZED)    ? "NOT_INITIALIZED"         \
          : (err == CUDSS_STATUS_ALLOC_FAILED)     ? "ALLOC_FAILED"            \
          : (err == CUDSS_STATUS_INVALID_VALUE)    ? "INVALID_VALUE"           \
          : (err == CUDSS_STATUS_NOT_SUPPORTED)    ? "NOT_SUPPORTED"           \
          : (err == CUDSS_STATUS_EXECUTION_FAILED) ? "EXECUTION_FAILED"        \
          : (err == CUDSS_STATUS_INTERNAL_ERROR)   ? "INTERNAL_ERROR"          \
                                                   : "UNKNOWN";                  \
      fprintf(stderr, "cuDSS error at %s:%d: status %d (%s)\n", __FILE__,      \
              __LINE__, (int)err, err_str);                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Contains data for linear system.
struct LinSysData {
  /** KKT matrix in CSC form (host). */
  QOCOCscMatrix* K;

  /** KKT matrix in CSR form (device) for cuDSS. */
  cudssMatrix_t K_csr;

  /** Permutation vector (not used for CUDA backend - kept for compatibility).
   */
  QOCOInt* p;

  /** Inverse of permutation vector (not used for CUDA backend - kept for
   * compatibility). */
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

  /** Device buffer for xyz (solution of KKT system) - used during solve phase.
   */
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

  /** CSR matrix arrays on GPU (persistent, updated directly) */
  QOCOInt* d_csr_row_ptr;
  QOCOInt* d_csr_row_end;
  QOCOInt* d_csr_col_ind;
  QOCOFloat* d_csr_val;

  /** Mappings on GPU for direct updates */
  QOCOInt* d_nt2kkt;
  QOCOInt* d_ntdiag2kkt;
  QOCOInt* d_PregtoKKT;
  QOCOInt* d_AttoKKT;
  QOCOInt* d_GttoKKT;

  /** CSC structure on GPU (for converting updates) */
  QOCOInt* d_csc_p;
  QOCOInt* d_csc_i;

  /** Flag to track if analysis has been done (per instance) */
  int analysis_done;
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
  CUDA_CHECK(cudaMemcpy(*csr_row_ptr, h_csr_row_ptr, (m + 1) * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_col_ind, h_csr_col_ind, nnz * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_val, h_csr_val, nnz * sizeof(QOCOFloat),
                        cudaMemcpyHostToDevice));

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
  linsys_data->PregtoKKT =
      data->P ? (QOCOInt*)qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  linsys_data->AttoKKT =
      (QOCOInt*)qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  linsys_data->GttoKKT =
      (QOCOInt*)qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  QOCOInt* nt2kkt_temp = (QOCOInt*)qoco_calloc(Wnnz, sizeof(QOCOInt));
  QOCOInt* ntdiag2kkt_temp = (QOCOInt*)qoco_calloc(data->m, sizeof(QOCOInt));
  QOCOInt* PregtoKKT_temp =
      data->P ? (QOCOInt*)qoco_calloc(get_nnz(data->P), sizeof(QOCOInt)) : NULL;
  QOCOInt* AttoKKT_temp =
      (QOCOInt*)qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  QOCOInt* GttoKKT_temp =
      (QOCOInt*)qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  // Construct KKT matrix (no permutation for CUDA backend)
  linsys_data->K = construct_kkt(
      data->P ? get_csc_matrix(data->P) : NULL, get_csc_matrix(data->A),
      get_csc_matrix(data->G), get_csc_matrix(data->At),
      get_csc_matrix(data->Gt), settings->kkt_static_reg, data->n, data->m,
      data->p, data->l, data->nsoc, data->q, PregtoKKT_temp, AttoKKT_temp,
      GttoKKT_temp, nt2kkt_temp, ntdiag2kkt_temp, Wnnz);

  // No AMD ordering or permutation - use KKT matrix directly
  // Set identity permutation (no-op)
  linsys_data->p = (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
  linsys_data->pinv =
      (QOCOInt*)qoco_malloc(linsys_data->K->n * sizeof(QOCOInt));
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

  // Convert KKT matrix from CSC (CPU) to CSR (GPU) for cuDSS - ONCE during
  // setup KKT matrix is formed on CPU in CSC, now convert and move to GPU Use
  // csc_to_csr_device helper function
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;

  csc_to_csr_device(linsys_data->K, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle);

  // Store CSR arrays on GPU persistently (will be updated directly, not
  // recreated)
  linsys_data->d_csr_row_ptr = csr_row_ptr;
  linsys_data->d_csr_col_ind = csr_col_ind;
  linsys_data->d_csr_val = csr_val;

// Create cuDSS matrix in CSR format
#ifdef HAVE_CUDSS
  // cuDSS CSR format: rowEnd can be NULL (cuDSS will compute it from rowStart)
  // The working example (simple.cpp) passes NULL for rowEnd
  // Store row_end for potential future use, but pass NULL to cuDSS
  QOCOInt* csr_row_end;
  CUDA_CHECK(cudaMalloc(&csr_row_end, Kn * sizeof(QOCOInt)));
  // Copy rowStart[1..n] to rowEnd[0..n-1]
  CUDA_CHECK(cudaMemcpy(csr_row_end, &csr_row_ptr[1], Kn * sizeof(QOCOInt),
                        cudaMemcpyDeviceToDevice));
  linsys_data->d_csr_row_end = csr_row_end;

  // Determine data types
  cudaDataType_t indexType = CUDA_R_32I; // QOCOInt is int32_t
  cudaDataType_t valueType_setup =
      (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;

  // KKT matrix is symmetric (upper triangular stored)
  // Use CUDSS_MTYPE_SPD (symmetric positive definite) like the working example
  // Pass NULL for rowEnd - cuDSS will compute it from rowStart
  CUDSS_CHECK(cudssMatrixCreateCsr(
      &linsys_data->K_csr, (int64_t)Kn, (int64_t)Kn,
      (int64_t)linsys_data->K->nnz, csr_row_ptr, NULL, csr_col_ind, csr_val,
      indexType, valueType_setup, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER,
      CUDSS_BASE_ZERO));

  // Run analysis once during setup (data structure already created above)
  linsys_data->analysis_done = 0; // Will be set to 1 after first analysis
#else
  // cuDSS not available - set to NULL (solve will use fallback)
  linsys_data->K_csr = NULL;
  linsys_data->d_rhs_matrix = NULL;
  linsys_data->d_xyz_matrix = NULL;
  linsys_data->d_csr_row_ptr = NULL;
  linsys_data->d_csr_row_end = NULL;
  linsys_data->d_csr_col_ind = NULL;
  linsys_data->d_csr_val = NULL;
#endif

  // Copy mappings to GPU for direct updates
  CUDA_CHECK(cudaMalloc(&linsys_data->d_nt2kkt, Wnnz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_ntdiag2kkt, data->m * sizeof(QOCOInt)));
  if (data->P && linsys_data->PregtoKKT) {
    CUDA_CHECK(cudaMalloc(&linsys_data->d_PregtoKKT,
                          get_nnz(data->P) * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMemcpy(linsys_data->d_PregtoKKT, linsys_data->PregtoKKT,
                          get_nnz(data->P) * sizeof(QOCOInt),
                          cudaMemcpyHostToDevice));
  }
  else {
    linsys_data->d_PregtoKKT = NULL;
  }
  CUDA_CHECK(
      cudaMalloc(&linsys_data->d_AttoKKT, get_nnz(data->A) * sizeof(QOCOInt)));
  CUDA_CHECK(
      cudaMalloc(&linsys_data->d_GttoKKT, get_nnz(data->G) * sizeof(QOCOInt)));

  CUDA_CHECK(cudaMemcpy(linsys_data->d_nt2kkt, linsys_data->nt2kkt,
                        Wnnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_ntdiag2kkt, linsys_data->ntdiag2kkt,
                        data->m * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_AttoKKT, linsys_data->AttoKKT,
                        get_nnz(data->A) * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_GttoKKT, linsys_data->GttoKKT,
                        get_nnz(data->G) * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));

  // Copy CSC structure to GPU (for converting updates in factor)
  CUDA_CHECK(cudaMalloc(&linsys_data->d_csc_p,
                        (linsys_data->K->n + 1) * sizeof(QOCOInt)));
  CUDA_CHECK(
      cudaMalloc(&linsys_data->d_csc_i, linsys_data->K->nnz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_csc_p, linsys_data->K->p,
                        (linsys_data->K->n + 1) * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_csc_i, linsys_data->K->i,
                        linsys_data->K->nnz * sizeof(QOCOInt),
                        cudaMemcpyHostToDevice));

  // Allocate device vectors for rhs and xyz
  // These will be used by the cuDSS dense matrix wrappers
  QOCOFloat* d_rhs_ptr = NULL;
  QOCOFloat* d_xyz_ptr = NULL;
  CUDA_CHECK(cudaMalloc(&d_rhs_ptr, Kn * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMalloc(&d_xyz_ptr, Kn * sizeof(QOCOFloat)));

  // Store in struct for cleanup
  linsys_data->d_xyzbuff1 = d_rhs_ptr; // Used for rhs buffer
  linsys_data->d_xyzbuff2 = d_xyz_ptr; // Used for xyz buffer

// Create dense matrix wrappers for solution and RHS vectors (column vectors)
// Note: d_rhs_matrix wraps d_xyzbuff1, d_xyz_matrix wraps d_xyzbuff2
#ifdef HAVE_CUDSS
  // Use valueType_setup declared above
  CUDSS_CHECK(cudssMatrixCreateDn(&linsys_data->d_rhs_matrix, (int64_t)Kn, 1,
                                  (int64_t)Kn, linsys_data->d_xyzbuff1,
                                  valueType_setup, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&linsys_data->d_xyz_matrix, (int64_t)Kn, 1,
                                  (int64_t)Kn, linsys_data->d_xyzbuff2,
                                  valueType_setup, CUDSS_LAYOUT_COL_MAJOR));
#else
  linsys_data->d_rhs_matrix = NULL;
  linsys_data->d_xyz_matrix = NULL;
#endif

  return linsys_data;
}

// CUDA kernel to update CSR values from updated CSC KKT matrix
// Maps CSC indices to CSR indices and updates values
__global__ void update_csr_from_csc_kernel(
    const QOCOFloat*
        csc_val,        // Updated CSC values (on GPU, copied from CPU KKT)
    QOCOFloat* csr_val, // CSR values to update (on GPU)
    const QOCOInt* csr_row_ptr, // CSR row pointers
    const QOCOInt* csr_col_ind, // CSR column indices
    const QOCOInt* csc_p,       // CSC column pointers (on GPU)
    const QOCOInt* csc_i,       // CSC row indices (on GPU)
    QOCOInt n,                  // Matrix dimension
    QOCOInt nnz)                // Number of nonzeros
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nnz)
    return;

  // For each CSR element at index idx, find corresponding CSC element
  // CSR: row = find row such that csr_row_ptr[row] <= idx < csr_row_ptr[row+1]
  //      col = csr_col_ind[idx]
  // CSC: find idx_csc such that csc_i[idx_csc] == row and column of idx_csc ==
  // col

  // Binary search for row
  QOCOInt row = 0;
  QOCOInt left = 0, right = n - 1;
  while (left <= right) {
    QOCOInt mid = (left + right) / 2;
    if (csr_row_ptr[mid] <= idx) {
      row = mid;
      left = mid + 1;
    }
    else {
      right = mid - 1;
    }
  }

  QOCOInt col = csr_col_ind[idx];

  // Find corresponding CSC element: search column col for row
  for (QOCOInt csc_idx = csc_p[col]; csc_idx < csc_p[col + 1]; csc_idx++) {
    if (csc_i[csc_idx] == row) {
      csr_val[idx] = csc_val[csc_idx];
      return;
    }
  }
}

static void cudss_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  // Update CSR matrix values on GPU directly - NO CPU-GPU transfers during
  // solve The KKT matrix was updated on CPU (in CSC), but we need to update GPU
  // CSR Copy updated CSC values to GPU, then update CSR values using a kernel

#ifdef HAVE_CUDSS
  // Copy updated CSC KKT matrix values to GPU (minimal transfer - only values,
  // not structure) This is the ONLY CPU-GPU transfer during solve iterations
  // (matrix values update)
  QOCOFloat* d_csc_val;
  CUDA_CHECK(cudaMalloc(&d_csc_val, linsys_data->K->nnz * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMemcpy(d_csc_val, linsys_data->K->x,
                        linsys_data->K->nnz * sizeof(QOCOFloat),
                        cudaMemcpyHostToDevice));

  // Update CSR values from CSC using CUDA kernel (CSC structure already on GPU
  // from setup)
  QOCOInt threadsPerBlock = 256;
  QOCOInt numBlocks =
      (linsys_data->K->nnz + threadsPerBlock - 1) / threadsPerBlock;
  update_csr_from_csc_kernel<<<numBlocks, threadsPerBlock>>>(
      d_csc_val, linsys_data->d_csr_val, linsys_data->d_csr_row_ptr,
      linsys_data->d_csr_col_ind, linsys_data->d_csc_p, linsys_data->d_csc_i,
      linsys_data->K->n, linsys_data->K->nnz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Free temporary CSC values buffer
  cudaFree(d_csc_val);

  // Update cuDSS matrix with new values
  CUDSS_CHECK(cudssMatrixSetValues(linsys_data->K_csr, linsys_data->d_csr_val));

  // Run factorization - data structure persists from setup
  // Analysis only needed once (first factorization), then just factorization
  if (!linsys_data->analysis_done) {
    cudssStatus_t status_analysis =
        cudssExecute(linsys_data->handle, CUDSS_PHASE_ANALYSIS,
                     linsys_data->config, linsys_data->data, linsys_data->K_csr,
                     linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix);
    if (status_analysis != CUDSS_STATUS_SUCCESS) {
      const char* err_str =
          (status_analysis == CUDSS_STATUS_NOT_INITIALIZED) ? "NOT_INITIALIZED"
          : (status_analysis == CUDSS_STATUS_ALLOC_FAILED)  ? "ALLOC_FAILED"
          : (status_analysis == CUDSS_STATUS_INVALID_VALUE) ? "INVALID_VALUE"
          : (status_analysis == CUDSS_STATUS_NOT_SUPPORTED) ? "NOT_SUPPORTED"
          : (status_analysis == CUDSS_STATUS_EXECUTION_FAILED)
              ? "EXECUTION_FAILED"
          : (status_analysis == CUDSS_STATUS_INTERNAL_ERROR) ? "INTERNAL_ERROR"
                                                             : "UNKNOWN";
      fprintf(stderr, "ERROR: cuDSS analysis failed with status %d (%s)\n",
              (int)status_analysis, err_str);
      exit(1);
    }
    linsys_data->analysis_done = 1;
  }

  cudssStatus_t status_factor =
      cudssExecute(linsys_data->handle, CUDSS_PHASE_FACTORIZATION,
                   linsys_data->config, linsys_data->data, linsys_data->K_csr,
                   linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix);
  if (status_factor != CUDSS_STATUS_SUCCESS) {
    const char* err_str =
        (status_factor == CUDSS_STATUS_NOT_INITIALIZED)    ? "NOT_INITIALIZED"
        : (status_factor == CUDSS_STATUS_ALLOC_FAILED)     ? "ALLOC_FAILED"
        : (status_factor == CUDSS_STATUS_INVALID_VALUE)    ? "INVALID_VALUE"
        : (status_factor == CUDSS_STATUS_NOT_SUPPORTED)    ? "NOT_SUPPORTED"
        : (status_factor == CUDSS_STATUS_EXECUTION_FAILED) ? "EXECUTION_FAILED"
        : (status_factor == CUDSS_STATUS_INTERNAL_ERROR)   ? "INTERNAL_ERROR"
                                                           : "UNKNOWN";
    fprintf(stderr, "ERROR: cuDSS factorization failed with status %d (%s)\n",
            (int)status_factor, err_str);
    exit(1);
  }
#else
// cuDSS not available - skip
#endif
}

// Helper function to map work->rhs and work->xyz to device buffers during solve
// phase
extern "C" {
void map_work_buffers_to_device(void* linsys_data_ptr, QOCOWorkspace* work)
{
  // During solve phase, functions constructing rhs/xyz will operate on device
  // buffers This function is called at start of solve phase Map
  // work->rhs->d_data and work->xyz->d_data to point to our device buffers so
  // that RHS construction and solution storage happen directly in cuDSS buffers
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  QOCOInt n = work->data->n + work->data->m + work->data->p;

  // Store original device pointers (if any) so we can restore them later
  // For now, just point d_data to our buffers
  if (work->rhs) {
    QOCOFloat* old_rhs = work->rhs->d_data;
    if (old_rhs && old_rhs != linsys_data->d_xyzbuff1) {
      CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff1, old_rhs,
                            n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
      cudaFree(old_rhs);
    }
    else if (!old_rhs) {
      CUDA_CHECK(cudaMemset(linsys_data->d_xyzbuff1, 0, n * sizeof(QOCOFloat)));
    }
    work->rhs->d_data = linsys_data->d_xyzbuff1;
  }

  if (work->xyz) {
    QOCOFloat* old_xyz = work->xyz->d_data;
    if (old_xyz && old_xyz != linsys_data->d_xyzbuff2) {
      CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff2, old_xyz,
                            n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
      cudaFree(old_xyz);
    }
    else if (!old_xyz) {
      CUDA_CHECK(cudaMemset(linsys_data->d_xyzbuff2, 0, n * sizeof(QOCOFloat)));
    }
    work->xyz->d_data = linsys_data->d_xyzbuff2;
  }
}

QOCOFloat* get_device_rhs(void* linsys_data_ptr)
{
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  // Return the device buffer where RHS should be constructed
  return linsys_data->d_xyzbuff1;
}

QOCOFloat* get_device_xyz(void* linsys_data_ptr)
{
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  // Return the device buffer where solution will be written
  return linsys_data->d_xyzbuff2;
}

void unmap_work_buffers_from_device(void* linsys_data_ptr, QOCOWorkspace* work)
{
  // Copy final results from device buffers back to host buffers
  // This is called AFTER solve completes, so it's the only CPU-GPU transfer
  // after solve
  LinSysData* linsys_data = (LinSysData*)linsys_data_ptr;
  QOCOInt n = work->data->n + work->data->m + work->data->p;

  // Restore original device pointers (reallocate them)
  if (work->rhs) {
    // Allocate new device buffer for work->rhs->d_data
    CUDA_CHECK(cudaMalloc(&work->rhs->d_data, n * sizeof(QOCOFloat)));
    // Copy from our buffer to the restored buffer
    CUDA_CHECK(cudaMemcpy(work->rhs->d_data, linsys_data->d_xyzbuff1,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
    // Copy to host
    CUDA_CHECK(cudaMemcpy(work->rhs->data, work->rhs->d_data,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  }

  if (work->xyz) {
    // Allocate new device buffer for work->xyz->d_data
    CUDA_CHECK(cudaMalloc(&work->xyz->d_data, n * sizeof(QOCOFloat)));
    // Copy solution from d_xyzbuff2 (where cuDSS wrote it) to restored buffer
    CUDA_CHECK(cudaMemcpy(work->xyz->d_data, linsys_data->d_xyzbuff2,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
    // Copy to host
    CUDA_CHECK(cudaMemcpy(work->xyz->data, work->xyz->d_data,
                          n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  }
}
}

static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOFloat* b, QOCOFloat* x, QOCOInt iter_ref_iters)
{
  QOCOInt n = linsys_data->K->n;
  (void)iter_ref_iters; // No iterative refinement for CUDA backend
  // b is copied to d_xyzbuff1 if provided
  // x will receive the solution from d_xyzbuff2 after cudssExecute

  // During solve phase, ALL data is on GPU - NO CPU-GPU transfers
  // work->rhs->d_data points to d_xyzbuff1 (mapped in
  // map_work_buffers_to_device) work->xyz->d_data points to d_xyzbuff2 (mapped
  // in map_work_buffers_to_device) RHS was constructed directly into
  // d_xyzbuff1, solution will be written to d_xyzbuff2

  // Ensure RHS buffer contains the contents of b (in case caller passed a
  // different pointer)
  if (b) {
    if (b != linsys_data->d_xyzbuff1) {
      cudaPointerAttributes attrs;
      cudaError_t err = cudaPointerGetAttributes(&attrs, b);
      if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff1, b, n * sizeof(QOCOFloat),
                              cudaMemcpyDeviceToDevice));
      }
      else {
        CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff1, b, n * sizeof(QOCOFloat),
                              cudaMemcpyHostToDevice));
      }
    }
  }
  else {
    CUDA_CHECK(cudaMemset(linsys_data->d_xyzbuff1, 0, n * sizeof(QOCOFloat)));
  }

  // Clear solution buffer (d_xyz_matrix points to d_xyzbuff2)
  CUDA_CHECK(cudaMemset(linsys_data->d_xyzbuff2, 0, n * sizeof(QOCOFloat)));

// cuDSS API signature: cudssExecute(handle, phase, config, data, matrix,
// solution, rhs) where solution is the output and rhs is the input Note:
// d_rhs_matrix points to d_xyzbuff1, d_xyz_matrix points to d_xyzbuff2
#ifdef HAVE_CUDSS

  cudssStatus_t status =
      cudssExecute(linsys_data->handle, CUDSS_PHASE_SOLVE, linsys_data->config,
                   linsys_data->data, linsys_data->K_csr,
                   linsys_data->d_xyz_matrix, linsys_data->d_rhs_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    const char* err_str =
        (status == CUDSS_STATUS_NOT_INITIALIZED)    ? "NOT_INITIALIZED"
        : (status == CUDSS_STATUS_ALLOC_FAILED)     ? "ALLOC_FAILED"
        : (status == CUDSS_STATUS_INVALID_VALUE)    ? "INVALID_VALUE"
        : (status == CUDSS_STATUS_NOT_SUPPORTED)    ? "NOT_SUPPORTED"
        : (status == CUDSS_STATUS_EXECUTION_FAILED) ? "EXECUTION_FAILED"
        : (status == CUDSS_STATUS_INTERNAL_ERROR)   ? "INTERNAL_ERROR"
                                                    : "UNKNOWN";
    fprintf(stderr, "ERROR: cuDSS solve failed with status %d (%s)\n",
            (int)status, err_str);
  }
  else {
    // Solution is now in d_xyzbuff2 (pointed to by d_xyz_matrix)
    // Copy solution from d_xyz_matrix (d_xyzbuff2) to x
    if (x) {
      cudaPointerAttributes attrs;
      cudaError_t err = cudaPointerGetAttributes(&attrs, x);
      if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        // x is on device - device-to-device copy
        CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyzbuff2, n * sizeof(QOCOFloat),
                              cudaMemcpyDeviceToDevice));
      }
      else {
        // x is on host - device-to-host copy
        CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyzbuff2, n * sizeof(QOCOFloat),
                              cudaMemcpyDeviceToHost));
      }
    }
  }

// Solution is now in d_xyzbuff2 (pointed to by d_xyz_matrix and
// work->xyz->d_data) Solution has been copied to x (if provided) after
// successful cudssExecute
#else
  // cuDSS not available - use fallback: copy solution from RHS (will fail
  // convergence but won't crash)
  CUDA_CHECK(cudaMemcpy(linsys_data->d_xyzbuff2, linsys_data->d_xyzbuff1,
                        n * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice));
  // Copy to x if provided
  if (x) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, x);
    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
      CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyzbuff2, n * sizeof(QOCOFloat),
                            cudaMemcpyDeviceToDevice));
    }
    else {
      CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyzbuff2, n * sizeof(QOCOFloat),
                            cudaMemcpyDeviceToHost));
    }
  }
#endif

  // During solve phase, solution stays on device in d_xyzbuff2 (and
  // work->xyz->d_data) It will be copied back to CPU only after solve completes
  // (in unmap_work_buffers_from_device)
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
  if (linsys_data->d_xyzbuff1)
    cudaFree(linsys_data->d_xyzbuff1);
  if (linsys_data->d_xyzbuff2)
    cudaFree(linsys_data->d_xyzbuff2);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  if (linsys_data->d_csr_row_ptr)
    cudaFree(linsys_data->d_csr_row_ptr);
  if (linsys_data->d_csr_row_end)
    cudaFree(linsys_data->d_csr_row_end);
  if (linsys_data->d_csr_col_ind)
    cudaFree(linsys_data->d_csr_col_ind);
  if (linsys_data->d_csr_val)
    cudaFree(linsys_data->d_csr_val);
  if (linsys_data->d_nt2kkt)
    cudaFree(linsys_data->d_nt2kkt);
  if (linsys_data->d_ntdiag2kkt)
    cudaFree(linsys_data->d_ntdiag2kkt);
  if (linsys_data->d_PregtoKKT)
    cudaFree(linsys_data->d_PregtoKKT);
  if (linsys_data->d_AttoKKT)
    cudaFree(linsys_data->d_AttoKKT);
  if (linsys_data->d_GttoKKT)
    cudaFree(linsys_data->d_GttoKKT);
  if (linsys_data->d_csc_p)
    cudaFree(linsys_data->d_csc_p);
  if (linsys_data->d_csc_i)
    cudaFree(linsys_data->d_csc_i);
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
