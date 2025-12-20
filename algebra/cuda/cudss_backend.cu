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
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

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

// Global function pointer structure
static CudaLibFuncs g_cuda_funcs = {0};
static void* g_cudss_handle = NULL;
static void* g_cusparse_handle = NULL;
static void* g_cublas_handle = NULL;
static int g_libs_loaded = 0;

// Global accessor for function pointers (for use in cuda_linalg.cu)
CudaLibFuncs* get_cuda_funcs(void) { return &g_cuda_funcs; }

// Load CUDA libraries using dlopen
int load_cuda_libraries(void)
{
  if (g_libs_loaded) {
    return 1; // Already loaded
  }

  // Load cuDSS
  g_cudss_handle = dlopen("libcudss.so", RTLD_LAZY);
  if (!g_cudss_handle) {
    g_cudss_handle = dlopen("libcudss.so.1", RTLD_LAZY);
  }
  if (!g_cudss_handle) {
    fprintf(stderr, "Failed to load cuDSS: %s\n", dlerror());
    return 0;
  }

  // Load cuSPARSE
  g_cusparse_handle = dlopen("libcusparse.so", RTLD_LAZY);
  if (!g_cusparse_handle) {
    g_cusparse_handle = dlopen("libcusparse.so.11", RTLD_LAZY);
  }
  if (!g_cusparse_handle) {
    g_cusparse_handle = dlopen("libcusparse.so.12", RTLD_LAZY);
  }
  if (!g_cusparse_handle) {
    fprintf(stderr, "Failed to load cuSPARSE: %s\n", dlerror());
    dlclose(g_cudss_handle);
    return 0;
  }

  // Load cuBLAS
  g_cublas_handle = dlopen("libcublas.so", RTLD_LAZY);
  if (!g_cublas_handle) {
    g_cublas_handle = dlopen("libcublas.so.11", RTLD_LAZY);
  }
  if (!g_cublas_handle) {
    g_cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY);
  }
  if (!g_cublas_handle) {
    fprintf(stderr, "Failed to load cuBLAS: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    return 0;
  }

  // Load cuDSS functions
  g_cuda_funcs.cudssCreate =
      (typeof(g_cuda_funcs.cudssCreate))dlsym(g_cudss_handle, "cudssCreate");
  g_cuda_funcs.cudssConfigCreate =
      (typeof(g_cuda_funcs.cudssConfigCreate))dlsym(g_cudss_handle,
                                                    "cudssConfigCreate");
  g_cuda_funcs.cudssDataCreate = (typeof(g_cuda_funcs.cudssDataCreate))dlsym(
      g_cudss_handle, "cudssDataCreate");
  g_cuda_funcs.cudssMatrixCreateCsr =
      (typeof(g_cuda_funcs.cudssMatrixCreateCsr))dlsym(g_cudss_handle,
                                                       "cudssMatrixCreateCsr");
  g_cuda_funcs.cudssExecute =
      (typeof(g_cuda_funcs.cudssExecute))dlsym(g_cudss_handle, "cudssExecute");
  g_cuda_funcs.cudssMatrixCreateDn =
      (typeof(g_cuda_funcs.cudssMatrixCreateDn))dlsym(g_cudss_handle,
                                                      "cudssMatrixCreateDn");
  g_cuda_funcs.cudssMatrixSetValues =
      (typeof(g_cuda_funcs.cudssMatrixSetValues))dlsym(g_cudss_handle,
                                                       "cudssMatrixSetValues");
  g_cuda_funcs.cudssMatrixDestroy =
      (typeof(g_cuda_funcs.cudssMatrixDestroy))dlsym(g_cudss_handle,
                                                     "cudssMatrixDestroy");
  g_cuda_funcs.cudssDataDestroy = (typeof(g_cuda_funcs.cudssDataDestroy))dlsym(
      g_cudss_handle, "cudssDataDestroy");
  g_cuda_funcs.cudssConfigDestroy =
      (typeof(g_cuda_funcs.cudssConfigDestroy))dlsym(g_cudss_handle,
                                                     "cudssConfigDestroy");
  g_cuda_funcs.cudssDestroy =
      (typeof(g_cuda_funcs.cudssDestroy))dlsym(g_cudss_handle, "cudssDestroy");

  if (!g_cuda_funcs.cudssCreate || !g_cuda_funcs.cudssConfigCreate ||
      !g_cuda_funcs.cudssDataCreate || !g_cuda_funcs.cudssMatrixCreateCsr ||
      !g_cuda_funcs.cudssExecute || !g_cuda_funcs.cudssMatrixCreateDn ||
      !g_cuda_funcs.cudssMatrixSetValues || !g_cuda_funcs.cudssMatrixDestroy ||
      !g_cuda_funcs.cudssDataDestroy || !g_cuda_funcs.cudssConfigDestroy ||
      !g_cuda_funcs.cudssDestroy) {
    fprintf(stderr, "Failed to resolve cuDSS symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    return 0;
  }

  // Load cuSPARSE functions
  g_cuda_funcs.cusparseCreate = (typeof(g_cuda_funcs.cusparseCreate))dlsym(
      g_cusparse_handle, "cusparseCreate");
  g_cuda_funcs.cusparseCreateMatDescr =
      (typeof(g_cuda_funcs.cusparseCreateMatDescr))dlsym(
          g_cusparse_handle, "cusparseCreateMatDescr");
  g_cuda_funcs.cusparseSetMatType =
      (typeof(g_cuda_funcs.cusparseSetMatType))dlsym(g_cusparse_handle,
                                                     "cusparseSetMatType");
  g_cuda_funcs.cusparseSetMatIndexBase =
      (typeof(g_cuda_funcs.cusparseSetMatIndexBase))dlsym(
          g_cusparse_handle, "cusparseSetMatIndexBase");
  g_cuda_funcs.cusparseDestroy = (typeof(g_cuda_funcs.cusparseDestroy))dlsym(
      g_cusparse_handle, "cusparseDestroy");
  g_cuda_funcs.cusparseDestroyMatDescr =
      (typeof(g_cuda_funcs.cusparseDestroyMatDescr))dlsym(
          g_cusparse_handle, "cusparseDestroyMatDescr");

  if (!g_cuda_funcs.cusparseCreate || !g_cuda_funcs.cusparseCreateMatDescr ||
      !g_cuda_funcs.cusparseSetMatType ||
      !g_cuda_funcs.cusparseSetMatIndexBase || !g_cuda_funcs.cusparseDestroy ||
      !g_cuda_funcs.cusparseDestroyMatDescr) {
    fprintf(stderr, "Failed to resolve cuSPARSE symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    dlclose(g_cublas_handle);
    return 0;
  }

  // Load cuBLAS functions
  g_cuda_funcs.cublasCreate = (typeof(g_cuda_funcs.cublasCreate))dlsym(
      g_cublas_handle, "cublasCreate_v2");
  if (!g_cuda_funcs.cublasCreate) {
    g_cuda_funcs.cublasCreate = (typeof(g_cuda_funcs.cublasCreate))dlsym(
        g_cublas_handle, "cublasCreate");
  }
  g_cuda_funcs.cublasDdot =
      (typeof(g_cuda_funcs.cublasDdot))dlsym(g_cublas_handle, "cublasDdot_v2");
  if (!g_cuda_funcs.cublasDdot) {
    g_cuda_funcs.cublasDdot =
        (typeof(g_cuda_funcs.cublasDdot))dlsym(g_cublas_handle, "cublasDdot");
  }
  g_cuda_funcs.cublasDestroy = (typeof(g_cuda_funcs.cublasDestroy))dlsym(
      g_cublas_handle, "cublasDestroy_v2");
  if (!g_cuda_funcs.cublasDestroy) {
    g_cuda_funcs.cublasDestroy = (typeof(g_cuda_funcs.cublasDestroy))dlsym(
        g_cublas_handle, "cublasDestroy");
  }
  g_cuda_funcs.cublasIdamin = (typeof(g_cuda_funcs.cublasIdamin))dlsym(
      g_cublas_handle, "cublasIdamin_v2");
  if (!g_cuda_funcs.cublasIdamin) {
    g_cuda_funcs.cublasIdamin = (typeof(g_cuda_funcs.cublasIdamin))dlsym(
        g_cublas_handle, "cublasIdamin");
  }
  g_cuda_funcs.cublasIdamax = (typeof(g_cuda_funcs.cublasIdamax))dlsym(
      g_cublas_handle, "cublasIdamax_v2");
  if (!g_cuda_funcs.cublasIdamax) {
    g_cuda_funcs.cublasIdamax = (typeof(g_cuda_funcs.cublasIdamax))dlsym(
        g_cublas_handle, "cublasIdamax");
  }

  if (!g_cuda_funcs.cublasCreate || !g_cuda_funcs.cublasDdot ||
      !g_cuda_funcs.cublasDestroy || !g_cuda_funcs.cublasIdamin ||
      !g_cuda_funcs.cublasIdamax) {
    fprintf(stderr, "Failed to resolve cuBLAS symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    dlclose(g_cublas_handle);
    return 0;
  }

  g_libs_loaded = 1;
  return 1;
}

// Contains data for linear system.
struct LinSysData {

  /** KKT matrix in CSR form (device) for cuDSS. */
  cudssMatrix_t K_csr;

  /** Number of rows/columns of KKT matrix. */
  QOCOInt Kn;

  /** cuDSS handle. */
  cudssHandle_t handle;

  /** cuDSS config. */
  cudssConfig_t config;

  /** cuDSS data. */
  cudssData_t data;

  /** Buffer of size n + m + p (device) - used for b and x vectors. */
  QOCOFloat* d_rhs_matrix_data;

  /** Buffer of size n + m + p (device). */
  QOCOFloat* d_xyz_matrix_data;

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

  /** Number of constraints (m) - stored for use in factor */
  QOCOInt m;

  /** cuSPARSE handle. */
  cusparseHandle_t cusparse_handle;

  /** Matrix description. */
  cusparseMatDescr_t descr;

  /** cuDSS dense matrix wrappers for solution and RHS vectors. */
  cudssMatrix_t d_rhs_matrix;
  cudssMatrix_t d_xyz_matrix;

  /** CSR data array (used for updating NT block and cudss_update_data) */
  QOCOFloat* d_csr_val;

  /** Mapping from NT block indices to CSR KKT matrix indices (device) */
  QOCOInt* d_nt2kktcsr;
  /** Mapping from NT diagonal indices to CSR KKT matrix indices (device) */
  QOCOInt* d_ntdiag2kktcsr;

  /** Device buffer for WtW values (persistent, reused across iterations) */
  QOCOFloat* d_WtW;

  /** Mapping from P, A, G indices to CSR KKT matrix indices (device) */
  QOCOInt* d_PregtoKKTcsr;
  QOCOInt* d_AttoKKTcsr;
  QOCOInt* d_GttoKKTcsr;
};

// Convert CSC to CSR on CPU and copy to GPU
static void csc_to_csr_device(const QOCOCscMatrix* csc, QOCOInt** csr_row_ptr,
                              QOCOInt** csr_col_ind, QOCOFloat** csr_val,
                              cusparseHandle_t handle,
                              QOCOInt** h_csr_row_ptr_out,
                              QOCOInt** h_csr_col_ind_out,
                              QOCOInt** csc2csr_out)
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

  // Fill CSR arrays and build CSC-to-CSR mapping
  QOCOInt* csc2csr = NULL;
  if (csc2csr_out) {
    csc2csr = (QOCOInt*)qoco_malloc(nnz * sizeof(QOCOInt));
  }

  for (QOCOInt col = 0; col < n; ++col) {
    for (QOCOInt csc_idx = csc->p[col]; csc_idx < csc->p[col + 1]; ++csc_idx) {
      QOCOInt row = csc->i[csc_idx];
      QOCOInt csr_idx = row_pos[row]++;
      h_csr_col_ind[csr_idx] = col;
      h_csr_val[csr_idx] = csc->x[csc_idx];
      if (csc2csr) {
        csc2csr[csc_idx] = csr_idx;
      }
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

  // Return host arrays and mapping if requested
  if (h_csr_row_ptr_out) {
    *h_csr_row_ptr_out = h_csr_row_ptr;
  }
  else {
    qoco_free(h_csr_row_ptr);
  }
  if (h_csr_col_ind_out) {
    *h_csr_col_ind_out = h_csr_col_ind;
  }
  else {
    qoco_free(h_csr_col_ind);
  }
  if (csc2csr_out) {
    *csc2csr_out = csc2csr;
  }
  else {
    if (csc2csr)
      qoco_free(csc2csr);
  }
  // Always free h_csr_val as it's not needed after device copy
  qoco_free(h_csr_val);
}

static LinSysData* cudss_setup(QOCOProblemData* data, QOCOSettings* settings,
                               QOCOInt Wnnz)
{
  // Load CUDA libraries dynamically
  if (!load_cuda_libraries()) {
    fprintf(stderr, "Failed to load CUDA libraries\n");
    return NULL;
  }

  LinSysData* linsys_data = (LinSysData*)qoco_malloc(sizeof(LinSysData));

  linsys_data->Kn = data->n + data->m + data->p;

  // Initialize cuDSS
  CUDSS_CHECK(g_cuda_funcs.cudssCreate(&linsys_data->handle));
  CUDSS_CHECK(g_cuda_funcs.cudssConfigCreate(&linsys_data->config));
  CUDSS_CHECK(
      g_cuda_funcs.cudssDataCreate(linsys_data->handle, &linsys_data->data));

  // Initialize cuSPARSE
  g_cuda_funcs.cusparseCreate(&linsys_data->cusparse_handle);
  g_cuda_funcs.cusparseCreateMatDescr(&linsys_data->descr);
  g_cuda_funcs.cusparseSetMatType(linsys_data->descr,
                                  CUSPARSE_MATRIX_TYPE_GENERAL);
  g_cuda_funcs.cusparseSetMatIndexBase(linsys_data->descr,
                                       CUSPARSE_INDEX_BASE_ZERO);

  // Allocate vector buffers
  CUDA_CHECK(cudaMalloc(&linsys_data->d_rhs_matrix_data,
                        sizeof(QOCOFloat) * linsys_data->Kn));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_xyz_matrix_data,
                        sizeof(QOCOFloat) * linsys_data->Kn));
  linsys_data->Wnnz = Wnnz;

  // Allocate memory for mappings to KKT matrix
  linsys_data->nt2kkt = (QOCOInt*)qoco_calloc(Wnnz, sizeof(QOCOInt));
  linsys_data->ntdiag2kkt = (QOCOInt*)qoco_calloc(data->m, sizeof(QOCOInt));
  linsys_data->PregtoKKT =
      (QOCOInt*)qoco_calloc(get_nnz(data->P), sizeof(QOCOInt));
  linsys_data->AttoKKT =
      (QOCOInt*)qoco_calloc(get_nnz(data->A), sizeof(QOCOInt));
  linsys_data->GttoKKT =
      (QOCOInt*)qoco_calloc(get_nnz(data->G), sizeof(QOCOInt));

  // Construct KKT matrix (no permutation for CUDA backend).
  // Need to set CPU mode to 1 so get_csc_matrix returns host pointers, since
  // construct_kkt is a host function.
  set_cpu_mode(1);
  QOCOCscMatrix* Kcsc = construct_kkt(
      get_csc_matrix(data->P), get_csc_matrix(data->A), get_csc_matrix(data->G),
      get_csc_matrix(data->At), get_csc_matrix(data->Gt),
      settings->kkt_static_reg, data->n, data->m, data->p, data->l, data->nsoc,
      get_data_vectori(data->q), linsys_data->PregtoKKT, linsys_data->AttoKKT,
      linsys_data->GttoKKT, linsys_data->nt2kkt, linsys_data->ntdiag2kkt, Wnnz);
  set_cpu_mode(0);

  // Convert KKT matrix from CSC (CPU) to CSR (GPU) for cuDSS
  QOCOInt* csr_row_ptr;
  QOCOInt* csr_col_ind;
  QOCOFloat* csr_val;
  QOCOInt* h_csr_row_ptr;
  QOCOInt* h_csr_col_ind;
  QOCOInt* csc2csr;

  csc_to_csr_device(Kcsc, &csr_row_ptr, &csr_col_ind, &csr_val,
                    linsys_data->cusparse_handle, &h_csr_row_ptr,
                    &h_csr_col_ind, &csc2csr);

  // Build nt2kktcsr and ntdiag2kktcsr mappings (CSR indices instead of CSC)
  QOCOInt* nt2kktcsr = NULL;
  if (Wnnz > 0) {
    nt2kktcsr = (QOCOInt*)qoco_malloc(Wnnz * sizeof(QOCOInt));
    for (QOCOInt i = 0; i < Wnnz; ++i) {
      nt2kktcsr[i] = csc2csr[linsys_data->nt2kkt[i]];
    }
  }
  QOCOInt* ntdiag2kktcsr = NULL;
  if (data->m > 0) {
    ntdiag2kktcsr = (QOCOInt*)qoco_malloc(data->m * sizeof(QOCOInt));
    QOCOInt diag_idx = 0;
    for (QOCOInt i = 0; i < data->m; ++i) {
      ntdiag2kktcsr[diag_idx] = csc2csr[linsys_data->ntdiag2kkt[i]];
      diag_idx++;
    }
  }

  // Build CSR mappings for P, A, G (convert from CSC indices to CSR indices)
  QOCOInt* PregtoKKTcsr = NULL;
  QOCOInt* AttoKKTcsr = NULL;
  QOCOInt* GttoKKTcsr = NULL;

  QOCOInt Pnnz = get_nnz(data->P);
  PregtoKKTcsr = (QOCOInt*)qoco_malloc(Pnnz * sizeof(QOCOInt));
  for (QOCOInt i = 0; i < Pnnz; ++i) {
    PregtoKKTcsr[i] = csc2csr[linsys_data->PregtoKKT[i]];
  }

  QOCOInt Annz = get_nnz(data->A);
  AttoKKTcsr = (QOCOInt*)qoco_malloc(Annz * sizeof(QOCOInt));
  for (QOCOInt i = 0; i < Annz; ++i) {
    AttoKKTcsr[i] = csc2csr[linsys_data->AttoKKT[data->AtoAt[i]]];
  }

  QOCOInt Gnnz = get_nnz(data->G);
  GttoKKTcsr = (QOCOInt*)qoco_malloc(Gnnz * sizeof(QOCOInt));
  for (QOCOInt i = 0; i < Gnnz; ++i) {
    GttoKKTcsr[i] = csc2csr[linsys_data->GttoKKT[data->GtoGt[i]]];
  }

  // Store CSR data array.
  linsys_data->d_csr_val = csr_val;

  // Determine data types
  cudaDataType_t indexType = CUDA_R_32I; // QOCOInt is int32_t
  cudaDataType_t valueType_setup =
      (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;

  // KKT matrix is symmetric (upper triangular stored)
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateCsr(
      &linsys_data->K_csr, (int64_t)linsys_data->Kn, (int64_t)linsys_data->Kn,
      (int64_t)Kcsc->nnz, csr_row_ptr, NULL, csr_col_ind, csr_val, indexType,
      valueType_setup, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER,
      CUDSS_BASE_ZERO));

  // Run analysis phase.
  CUDSS_CHECK(g_cuda_funcs.cudssExecute(
      linsys_data->handle, CUDSS_PHASE_ANALYSIS, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));

  // Free CSR structure arrays - cuDSS uses them during analysis
  cudaFree(csr_row_ptr);
  cudaFree(csr_col_ind);

  // Allocate and copy nt2kktcsr and ntdiag2kktcsr to device
  if (Wnnz > 0) {
    CUDA_CHECK(cudaMalloc(&linsys_data->d_nt2kktcsr, Wnnz * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMemcpy(linsys_data->d_nt2kktcsr, nt2kktcsr,
                          Wnnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
    // Allocate persistent buffer for WtW values (reused across iterations)
    CUDA_CHECK(cudaMalloc(&linsys_data->d_WtW, Wnnz * sizeof(QOCOFloat)));
  }
  else {
    linsys_data->d_nt2kktcsr = NULL;
    linsys_data->d_WtW = NULL;
  }
  // Only allocate ntdiag2kktcsr if m > 0 (i.e. if there are conic constraints)
  if (data->m > 0) {
    CUDA_CHECK(
        cudaMalloc(&linsys_data->d_ntdiag2kktcsr, data->m * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMemcpy(linsys_data->d_ntdiag2kktcsr, ntdiag2kktcsr,
                          data->m * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  }
  else {
    linsys_data->d_ntdiag2kktcsr = NULL;
  }

  // Allocate and copy P, A, G CSR mappings to device
  if (PregtoKKTcsr) {
    QOCOInt Pnnz = get_nnz(data->P);
    CUDA_CHECK(
        cudaMalloc(&linsys_data->d_PregtoKKTcsr, Pnnz * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMemcpy(linsys_data->d_PregtoKKTcsr, PregtoKKTcsr,
                          Pnnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  }
  else {
    linsys_data->d_PregtoKKTcsr = NULL;
  }

  CUDA_CHECK(cudaMalloc(&linsys_data->d_AttoKKTcsr, Annz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_AttoKKTcsr, AttoKKTcsr,
                        Annz * sizeof(QOCOInt), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&linsys_data->d_GttoKKTcsr, Gnnz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_GttoKKTcsr, GttoKKTcsr,
                        Gnnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));

  // Free temporary host arrays
  qoco_free(nt2kktcsr);
  qoco_free(ntdiag2kktcsr);
  qoco_free(PregtoKKTcsr);
  qoco_free(AttoKKTcsr);
  qoco_free(GttoKKTcsr);
  qoco_free(h_csr_row_ptr);
  qoco_free(h_csr_col_ind);
  qoco_free(csc2csr);

  // Create dense matrix wrappers for solution and RHS vectors (column vectors)
  // Note: d_rhs_matrix wraps d_rhs_matrix_data, d_xyz_matrix wraps
  // d_xyz_matrix_data
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_rhs_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_rhs_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_xyz_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_xyz_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));

  return linsys_data;
}

// CUDA kernel to directly update CSR values for NT blocks
__global__ void
update_csr_nt_blocks_kernel(const QOCOFloat* WtW, // NT block values (on GPU)
                            QOCOFloat* csr_val, // CSR values to update (on GPU)
                            const QOCOInt* nt2kktcsr,
                            const QOCOInt* ntdiag2kktcsr,
                            QOCOFloat kkt_static_reg, QOCOInt Wnnz, QOCOInt m)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Update NT block values
  if (idx < Wnnz) {
    QOCOInt csr_idx = nt2kktcsr[idx];
    csr_val[csr_idx] = -WtW[idx];
  }
}

// CUDA kernel to update NT diagonal regularization
__global__ void
update_csr_nt_diag_kernel(QOCOFloat* csr_val, // CSR values to update (on GPU)
                          const QOCOInt* ntdiag2kktcsr,
                          QOCOFloat kkt_static_reg, QOCOInt m)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    QOCOInt csr_idx = ntdiag2kktcsr[idx];
    csr_val[csr_idx] -= kkt_static_reg;
  }
}

// CUDA kernel to update CSR values for P, A, G matrices
__global__ void update_csr_matrix_data_kernel(
    const QOCOFloat* matrix_val, // New matrix values (on GPU)
    QOCOFloat* csr_val,          // CSR values to update (on GPU)
    const QOCOInt* mat2kktcsr, QOCOInt nnz)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz) {
    QOCOInt csr_idx = mat2kktcsr[idx];
    csr_val[csr_idx] = matrix_val[idx];
  }
}

static void cudss_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  CUDSS_CHECK(g_cuda_funcs.cudssExecute(
      linsys_data->handle, CUDSS_PHASE_FACTORIZATION, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));
}

static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOFloat* b, QOCOFloat* x, QOCOInt iter_ref_iters)
{
  (void)iter_ref_iters; // No iterative refinement for CUDA backend

  // Copy b from CPU to GPU.
  CUDA_CHECK(cudaMemcpy(linsys_data->d_rhs_matrix_data, b,
                        linsys_data->Kn * sizeof(QOCOFloat),
                        cudaMemcpyDeviceToDevice));

  // Clear solution buffer (d_xyz_matrix points to d_xyz_matrix_data)
  CUDA_CHECK(cudaMemset(linsys_data->d_xyz_matrix_data, 0,
                        linsys_data->Kn * sizeof(QOCOFloat)));

  // d_rhs_matrix points to d_rhs_matrix_data, d_xyz_matrix points to
  // d_xyz_matrix_data
  CUDSS_CHECK(g_cuda_funcs.cudssExecute(
      linsys_data->handle, CUDSS_PHASE_SOLVE, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));

  // Copy x from GPU to CPU.
  CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyz_matrix_data,
                        linsys_data->Kn * sizeof(QOCOFloat),
                        cudaMemcpyDeviceToDevice));
}

static void cudss_update_nt(LinSysData* linsys_data, QOCOFloat* WtW,
                            QOCOFloat kkt_static_reg, QOCOInt m)
{
  // Update CSR matrix values on GPU directly for NT blocks
  if (linsys_data->Wnnz > 0 && linsys_data->d_nt2kktcsr) {
    // Copy WtW to device from host
    CUDA_CHECK(cudaMemcpy(linsys_data->d_WtW, WtW,
                          linsys_data->Wnnz * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToDevice));

    // Update NT blocks directly in CSR
    QOCOInt threadsPerBlock = 256;
    QOCOInt numBlocks_nt =
        (linsys_data->Wnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_nt_blocks_kernel<<<numBlocks_nt, threadsPerBlock>>>(
        linsys_data->d_WtW, linsys_data->d_csr_val, linsys_data->d_nt2kktcsr,
        linsys_data->d_ntdiag2kktcsr, kkt_static_reg, linsys_data->Wnnz, m);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update diagonal regularization separately
  if (m > 0 && linsys_data->d_ntdiag2kktcsr) {
    QOCOInt threadsPerBlock = 256;
    QOCOInt numBlocks_diag = (m + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_nt_diag_kernel<<<numBlocks_diag, threadsPerBlock>>>(
        linsys_data->d_csr_val, linsys_data->d_ntdiag2kktcsr, kkt_static_reg,
        m);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(linsys_data->K_csr,
                                                linsys_data->d_csr_val));
  CUDA_CHECK(cudaDeviceSynchronize());
}

static void cudss_update_data(LinSysData* linsys_data, QOCOProblemData* data)
{
  // Update P, A, G directly in CSR matrix on GPU
  QOCOInt threadsPerBlock = 256;

  // Update P in CSR matrix
  if (data->P && linsys_data->d_PregtoKKTcsr) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    QOCOInt Pnnz = get_nnz(data->P);

    // Copy P values to device
    QOCOFloat* d_Pval;
    CUDA_CHECK(cudaMalloc(&d_Pval, Pnnz * sizeof(QOCOFloat)));
    CUDA_CHECK(cudaMemcpy(d_Pval, Pcsc->x, Pnnz * sizeof(QOCOFloat),
                          cudaMemcpyHostToDevice));

    // Update CSR values directly
    QOCOInt numBlocks = (Pnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocks, threadsPerBlock>>>(
        d_Pval, linsys_data->d_csr_val, linsys_data->d_PregtoKKTcsr, Pnnz);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_Pval);
  }

  // Update A in CSR matrix
  QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
  QOCOInt Annz = get_nnz(data->A);

  // Copy A values to device
  QOCOFloat* d_Aval;
  CUDA_CHECK(cudaMalloc(&d_Aval, Annz * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMemcpy(d_Aval, Acsc->x, Annz * sizeof(QOCOFloat),
                        cudaMemcpyHostToDevice));

  // Update CSR values directly
  QOCOInt numBlocksA = (Annz + threadsPerBlock - 1) / threadsPerBlock;
  update_csr_matrix_data_kernel<<<numBlocksA, threadsPerBlock>>>(
      d_Aval, linsys_data->d_csr_val, linsys_data->d_AttoKKTcsr, Annz);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_Aval);

  // Update G in CSR matrix
  QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
  QOCOInt Gnnz = get_nnz(data->G);

  // Copy G values to device
  QOCOFloat* d_Gval;
  CUDA_CHECK(cudaMalloc(&d_Gval, Gnnz * sizeof(QOCOFloat)));
  CUDA_CHECK(cudaMemcpy(d_Gval, Gcsc->x, Gnnz * sizeof(QOCOFloat),
                        cudaMemcpyHostToDevice));

  // Update CSR values directly
  QOCOInt numBlocksG = (Gnnz + threadsPerBlock - 1) / threadsPerBlock;
  update_csr_matrix_data_kernel<<<numBlocksG, threadsPerBlock>>>(
      d_Gval, linsys_data->d_csr_val, linsys_data->d_GttoKKTcsr, Gnnz);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_Gval);

  CUDA_CHECK(cudaDeviceSynchronize());
}

static void cudss_cleanup(LinSysData* linsys_data)
{
  if (g_libs_loaded) {
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->K_csr);
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->d_rhs_matrix);
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->d_xyz_matrix);
    g_cuda_funcs.cudssDataDestroy(linsys_data->handle, linsys_data->data);
    g_cuda_funcs.cudssConfigDestroy(linsys_data->config);
    g_cuda_funcs.cudssDestroy(linsys_data->handle);
    g_cuda_funcs.cusparseDestroy(linsys_data->cusparse_handle);
    g_cuda_funcs.cusparseDestroyMatDescr(linsys_data->descr);
  }
  cudaFree(linsys_data->d_rhs_matrix_data);
  cudaFree(linsys_data->d_xyz_matrix_data);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  cudaFree(linsys_data->d_csr_val);
  cudaFree(linsys_data->d_nt2kktcsr);
  cudaFree(linsys_data->d_ntdiag2kktcsr);
  cudaFree(linsys_data->d_WtW);
  cudaFree(linsys_data->d_PregtoKKTcsr);
  cudaFree(linsys_data->d_AttoKKTcsr);
  cudaFree(linsys_data->d_GttoKKTcsr);
  qoco_free(linsys_data);
}

static const char* cudss_name() { return "cuda/cuDSS"; }

LinSysBackend backend = {.linsys_name = cudss_name,
                         .linsys_setup = cudss_setup,
                         .linsys_update_nt = cudss_update_nt,
                         .linsys_update_data = cudss_update_data,
                         .linsys_factor = cudss_factor,
                         .linsys_solve = cudss_solve,
                         .linsys_cleanup = cudss_cleanup};
