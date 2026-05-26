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
#ifdef __cplusplus
extern "C" {
#endif
#include "equilibration.h"
#include "input_validation.h"
#include "qoco_utils.h"
#ifdef __cplusplus
}
#endif
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct {
  double serial_factor_sec;
  double serial_solve_sec;
  long long serial_factor_calls;
  long long serial_solve_calls;
  double batch_factor_sec;
  double batch_solve_sec;
  long long batch_factor_calls;
  long long batch_solve_calls;
} QOCOCudaLinsysTiming;

static QOCOCudaLinsysTiming g_linsys_timing = {0};
static int g_linsys_timing_enabled = 0;

static double qoco_cuda_now_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

extern "C" void qoco_cuda_linsys_timing_reset(void)
{
  memset(&g_linsys_timing, 0, sizeof(g_linsys_timing));
  g_linsys_timing_enabled = 1;
}

extern "C" void qoco_cuda_linsys_timing_set_enabled(int enabled)
{
  g_linsys_timing_enabled = enabled != 0;
}

extern "C" void qoco_cuda_linsys_timing_get(QOCOCudaLinsysTiming* timing)
{
  if (timing) {
    *timing = g_linsys_timing;
  }
}

static void qoco_cuda_timed_cudss_execute(
    cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t config,
    cudssData_t data, cudssMatrix_t matA, cudssMatrix_t matB,
    cudssMatrix_t matC, unsigned char is_batch)
{
  if (!g_linsys_timing_enabled ||
      (phase != CUDSS_PHASE_FACTORIZATION && phase != CUDSS_PHASE_SOLVE)) {
    CUDSS_CHECK(g_cuda_funcs.cudssExecute(handle, phase, config, data, matA,
                                          matB, matC));
    return;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  const double start_sec = qoco_cuda_now_sec();
  CUDSS_CHECK(g_cuda_funcs.cudssExecute(handle, phase, config, data, matA, matB,
                                        matC));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsed_sec = qoco_cuda_now_sec() - start_sec;

  if (is_batch) {
    if (phase == CUDSS_PHASE_FACTORIZATION) {
      g_linsys_timing.batch_factor_sec += elapsed_sec;
      g_linsys_timing.batch_factor_calls++;
    }
    else {
      g_linsys_timing.batch_solve_sec += elapsed_sec;
      g_linsys_timing.batch_solve_calls++;
    }
  }
  else {
    if (phase == CUDSS_PHASE_FACTORIZATION) {
      g_linsys_timing.serial_factor_sec += elapsed_sec;
      g_linsys_timing.serial_factor_calls++;
    }
    else {
      g_linsys_timing.serial_solve_sec += elapsed_sec;
      g_linsys_timing.serial_solve_calls++;
    }
  }
}

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
  g_cuda_funcs.cudssConfigSet = (typeof(g_cuda_funcs.cudssConfigSet))dlsym(
      g_cudss_handle, "cudssConfigSet");
  g_cuda_funcs.cudssMatrixCreateCsr =
      (typeof(g_cuda_funcs.cudssMatrixCreateCsr))dlsym(g_cudss_handle,
                                                       "cudssMatrixCreateCsr");
  g_cuda_funcs.cudssMatrixCreateBatchCsr =
      (typeof(g_cuda_funcs.cudssMatrixCreateBatchCsr))dlsym(
          g_cudss_handle, "cudssMatrixCreateBatchCsr");
  g_cuda_funcs.cudssExecute =
      (typeof(g_cuda_funcs.cudssExecute))dlsym(g_cudss_handle, "cudssExecute");
  g_cuda_funcs.cudssMatrixCreateDn =
      (typeof(g_cuda_funcs.cudssMatrixCreateDn))dlsym(g_cudss_handle,
                                                      "cudssMatrixCreateDn");
  g_cuda_funcs.cudssMatrixCreateBatchDn =
      (typeof(g_cuda_funcs.cudssMatrixCreateBatchDn))dlsym(
          g_cudss_handle, "cudssMatrixCreateBatchDn");
  g_cuda_funcs.cudssMatrixSetValues =
      (typeof(g_cuda_funcs.cudssMatrixSetValues))dlsym(g_cudss_handle,
                                                       "cudssMatrixSetValues");
  g_cuda_funcs.cudssMatrixSetBatchValues =
      (typeof(g_cuda_funcs.cudssMatrixSetBatchValues))dlsym(
          g_cudss_handle, "cudssMatrixSetBatchValues");
  g_cuda_funcs.cudssMatrixSetBatchCsrPointers =
      (typeof(g_cuda_funcs.cudssMatrixSetBatchCsrPointers))dlsym(
          g_cudss_handle, "cudssMatrixSetBatchCsrPointers");
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
      !g_cuda_funcs.cudssDestroy || !g_cuda_funcs.cudssConfigSet) {
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

  /** Static regularization for the (1,1) P block. */
  QOCOFloat kkt_static_reg_P;

  /** Static regularization for the (2,2) A block. */
  QOCOFloat kkt_static_reg_A;

  /** Static regularization for the (3,3) G block. */
  QOCOFloat kkt_static_reg_G;

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

struct CudaBatchLinSysData {
  QOCOInt batch_count;
  QOCOInt Kn;
  QOCOInt nnz;

  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t data;

  cudssMatrix_t K_batch;
  cudssMatrix_t rhs_batch;
  cudssMatrix_t xyz_batch;

  QOCOInt* d_csr_row_ptr;
  QOCOInt* d_csr_col_ind;

  void** h_csr_row_ptrs;
  void** h_csr_col_inds;
  void** h_csr_values;
  void** h_rhs_values;
  void** h_xyz_values;

  void** d_csr_row_ptrs;
  void** d_csr_col_inds;
  void** d_csr_values;
  void** d_rhs_values;
  void** d_xyz_values;

  QOCOInt* h_nrows;
  QOCOInt* h_ncols;
  QOCOInt* h_nnz;
  QOCOInt* h_nrhs;
  QOCOInt* h_ld;
};

extern "C" void qoco_cuda_batch_cleanup(QOCOBatchSolver* batch);

// Convert CSC to CSR on CPU and copy to GPU
static void csc_to_csr_device(const QOCOCscMatrix* csc, QOCOInt** csr_row_ptr,
                              QOCOInt** csr_col_ind, QOCOFloat** csr_val,
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
  int value = 0;
  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
                                          CUDSS_CONFIG_USE_SUPERPANELS,
                                          (void*)&value, sizeof(int)));

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
  linsys_data->kkt_static_reg_P = settings->kkt_static_reg_P;
  linsys_data->kkt_static_reg_A = settings->kkt_static_reg_A;
  linsys_data->kkt_static_reg_G = settings->kkt_static_reg_G;

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
      settings->kkt_static_reg_A, data->n, data->m, data->p, data->l, data->nsoc,
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
                    &h_csr_row_ptr, &h_csr_col_ind, &csc2csr);

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

  // Create dense matrix wrappers for solution and RHS vectors (column vectors).
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_rhs_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_rhs_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_xyz_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_xyz_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));

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

  return linsys_data;
}

// CUDA kernel to directly update CSR values for NT blocks
__global__ void
update_csr_nt_blocks_kernel(const QOCOFloat* WtW, // NT block values (on GPU)
                            QOCOFloat* csr_val, // CSR values to update (on GPU)
                            const QOCOInt* nt2kktcsr,
                            QOCOInt Wnnz)
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
                          QOCOFloat kkt_static_reg_G, QOCOInt m)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    QOCOInt csr_idx = ntdiag2kktcsr[idx];
    csr_val[csr_idx] -= kkt_static_reg_G;
  }
}

__global__ void set_nt_zero_kernel(double* Kx, const int* nt2kkt, int Wnnz,
                                   int m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < Wnnz) {
    Kx[nt2kkt[tid]] = 0.0;
  }
}

__global__ void set_nt_identity_kernel(double* Kx, const int* nt2kkt,
                                       const int* ntdiag2kkt, int Wnnz, int m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m) {
    Kx[ntdiag2kkt[tid]] = -1.0;
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
  (void)n;
  (void)kkt_dynamic_reg;

  qoco_cuda_timed_cudss_execute(
      linsys_data->handle, CUDSS_PHASE_FACTORIZATION, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix, 0);
}

static void cudss_solve_system(LinSysData* linsys_data, const QOCOFloat* rhs,
                               QOCOFloat* sol)
{
  CUDA_CHECK(cudaMemcpy(linsys_data->d_rhs_matrix_data, rhs,
                        linsys_data->Kn * sizeof(QOCOFloat),
                        cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaMemset(linsys_data->d_xyz_matrix_data, 0,
                        linsys_data->Kn * sizeof(QOCOFloat)));

  qoco_cuda_timed_cudss_execute(
      linsys_data->handle, CUDSS_PHASE_SOLVE, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix, 0);

  if (sol != linsys_data->d_xyz_matrix_data) {
    CUDA_CHECK(cudaMemcpy(sol, linsys_data->d_xyz_matrix_data,
                          linsys_data->Kn * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToDevice));
  }
}

/**
 * @brief Computes norm(K_true*x - b, inf) on device.
 *
 * The factorization uses the statically regularized KKT matrix. The residual
 * used for iterative refinement is computed against the unregularized KKT
 * product, matching the builtin backend behavior. The residual b - K_true*x is
 * written to residual_scratch.
 */
static QOCOFloat compute_linsys_residual(LinSysData* linsys_data,
                                         QOCOWorkspace* work,
                                         const QOCOFloat* b,
                                         const QOCOFloat* x,
                                         QOCOFloat* residual_scratch)
{
  QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
  QOCOInt* Wsoc_idx = get_data_vectori(work->Wsoc_idx);
  QOCOInt* soc_idx = get_data_vectori(work->soc_idx);
  QOCOFloat* xbuff = get_data_vectorf(work->xbuff);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOInt n = work->data->n;
  QOCOInt N = linsys_data->Kn;

  // d_rhs_matrix_data is scratch here; cudss_solve_system overwrites it before
  // every cuDSS solve.
  kkt_multiply((QOCOFloat*)x, linsys_data->d_rhs_matrix_data, work->data, Wfull,
               Wsoc_idx, soc_idx, xbuff, ubuff1, ubuff2);

  // data->P stores P + eps_P * I, so remove the P regularization from the
  // product before measuring the true KKT residual.
  qoco_axpy(x, linsys_data->d_rhs_matrix_data, linsys_data->d_rhs_matrix_data,
            -linsys_data->kkt_static_reg_P, n);

  // residual_scratch = b - K_true*x.
  qoco_axpy(linsys_data->d_rhs_matrix_data, b, residual_scratch, -1.0, N);

  return inf_norm(residual_scratch, N);
}

#ifdef QOCO_LOGGING
static void log_linsys_error(LinSysData* linsys_data, QOCOWorkspace* work,
                             const QOCOFloat* b, const QOCOFloat* x,
                             QOCOFloat* residual_scratch, const char* label,
                             FILE* f)
{
  QOCOFloat res =
      compute_linsys_residual(linsys_data, work, b, x, residual_scratch);
  fprintf(f, "  (%s): %.4e\n", label, res);
}
#endif

static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOVectorf* b_vec, QOCOVectorf* x_vec,
                        QOCOFloat ir_tol, QOCOInt max_ir_iters)
{
  QOCOFloat* x = get_data_vectorf(x_vec);
  QOCOFloat* b = get_data_vectorf(b_vec);
  QOCOFloat* residual = linsys_data->d_xyz_matrix_data;

  // Initial solve. Store the current solution in x; d_xyz_matrix_data remains
  // available as residual/correction scratch after this copy.
  cudss_solve_system(linsys_data, b, x);

#ifdef QOCO_LOGGING
  FILE* log_f = fopen("qoco_log.txt", "a");
  if (log_f) {
    log_linsys_error(linsys_data, work, b, x, residual, "initial solve",
                     log_f);
  }
#endif

  QOCOFloat* best_sol = get_data_vectorf(work->xyzbuff1);
  QOCOFloat best_res = compute_linsys_residual(linsys_data, work, b, x,
                                               residual);
  copy_arrayf(x, best_sol, linsys_data->Kn);

  QOCOInt ir_count = 0;
  QOCOFloat res = best_res;

  for (QOCOInt i = 0; i < max_ir_iters; ++i) {
    if (res < ir_tol) {
      break;
    }

    // residual currently holds b - K_true*x. Solve K_reg*dx = residual.
    cudss_solve_system(linsys_data, residual, linsys_data->d_xyz_matrix_data);

    // x_new = x_old + dx.
    qoco_axpy(linsys_data->d_xyz_matrix_data, x, x, 1.0, linsys_data->Kn);

    QOCOFloat new_res = compute_linsys_residual(linsys_data, work, b, x,
                                                residual);

#ifdef QOCO_LOGGING
    if (log_f) {
      fprintf(log_f, "  (refinement): %.4e\n", new_res);
    }
#endif

    if (new_res >= best_res) {
      copy_arrayf(best_sol, x, linsys_data->Kn);
      break;
    }

    ir_count++;
    best_res = new_res;
    copy_arrayf(x, best_sol, linsys_data->Kn);
    res = new_res;
  }

  work->ir_iters += ir_count;

#ifdef QOCO_LOGGING
  if (log_f)
    fclose(log_f);
#endif
}

void cudss_set_nt_identity(LinSysData* linsys_data, QOCOInt m)
{
  int Wnnz = linsys_data->Wnnz;

  int N = max(Wnnz, m);
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  if (m > 0) {
    set_nt_zero_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_nt2kktcsr, Wnnz, m);

    CUDA_CHECK(cudaDeviceSynchronize());

    set_nt_identity_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_nt2kktcsr,
        linsys_data->d_ntdiag2kktcsr, Wnnz, m);
    CUDA_CHECK(cudaGetLastError());

    update_csr_nt_diag_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_ntdiag2kktcsr,
        linsys_data->kkt_static_reg_G, m);
    CUDA_CHECK(cudaGetLastError());
    CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(linsys_data->K_csr,
                                                  linsys_data->d_csr_val));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

static void cudss_update_nt(LinSysData* linsys_data, QOCOVectorf* WtW_vec,
                            QOCOFloat kkt_static_reg_G, QOCOInt m)
{
  QOCOFloat* WtW = get_data_vectorf(WtW_vec);
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
        linsys_data->Wnnz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update diagonal regularization separately
  if (m > 0 && linsys_data->d_ntdiag2kktcsr) {
    QOCOInt threadsPerBlock = 256;
    QOCOInt numBlocks_diag = (m + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_nt_diag_kernel<<<numBlocks_diag, threadsPerBlock>>>(
        linsys_data->d_csr_val, linsys_data->d_ntdiag2kktcsr, kkt_static_reg_G,
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

    // Update CSR KKT matrix
    QOCOInt numBlocks = (Pnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocks, threadsPerBlock>>>(
        Pcsc->x, linsys_data->d_csr_val, linsys_data->d_PregtoKKTcsr, Pnnz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update A in CSR matrix
  if (data->p > 0) {
    QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
    QOCOInt Annz = get_nnz(data->A);

    // Update CSR KKT matrix
    QOCOInt numBlocksA = (Annz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocksA, threadsPerBlock>>>(
        Acsc->x, linsys_data->d_csr_val, linsys_data->d_AttoKKTcsr, Annz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update G in CSR matrix
  if (data->m > 0) {

    QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
    QOCOInt Gnnz = get_nnz(data->G);

    // Update CSR KKT matrix
    QOCOInt numBlocksG = (Gnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocksG, threadsPerBlock>>>(
        Gcsc->x, linsys_data->d_csr_val, linsys_data->d_GttoKKTcsr, Gnnz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
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

static unsigned char cuda_batch_symbols_available(void)
{
  return g_cuda_funcs.cudssMatrixCreateBatchCsr &&
         g_cuda_funcs.cudssMatrixCreateBatchDn;
}

static void cuda_batch_refresh_value_pointers(QOCOBatchSolver* batch,
                                              CudaBatchLinSysData* batch_data)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    LinSysData* linsys_data = (LinSysData*)batch->solvers[item]->linsys_data;
    batch_data->h_csr_values[item] = (void*)linsys_data->d_csr_val;
    batch_data->h_rhs_values[item] = (void*)linsys_data->d_rhs_matrix_data;
    batch_data->h_xyz_values[item] = (void*)linsys_data->d_xyz_matrix_data;
  }

  CUDA_CHECK(cudaMemcpy(batch_data->d_csr_values, batch_data->h_csr_values,
                        batch_data->batch_count * sizeof(void*),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch_data->d_rhs_values, batch_data->h_rhs_values,
                        batch_data->batch_count * sizeof(void*),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch_data->d_xyz_values, batch_data->h_xyz_values,
                        batch_data->batch_count * sizeof(void*),
                        cudaMemcpyHostToDevice));
}

static QOCOInt cuda_batch_validate_solvers(QOCOBatchSolver* batch)
{
  QOCOSolver* first = batch->solvers[0];
  QOCOProblemData* first_data = first->work->data;
  QOCOInt first_Pnnz = get_nnz(first_data->P);
  QOCOInt first_Annz = get_nnz(first_data->A);
  QOCOInt first_Gnnz = get_nnz(first_data->G);

  for (QOCOInt item = 0; item < batch->batch_count; ++item) {
    QOCOSolver* solver = batch->solvers[item];
    if (!solver || !solver->work || !solver->work->data ||
        !solver->linsys_data) {
      return QOCO_DATA_VALIDATION_ERROR;
    }

    QOCOProblemData* data = solver->work->data;
    if (data->n != first_data->n || data->m != first_data->m ||
        data->p != first_data->p || data->l != first_data->l ||
        data->nsoc != first_data->nsoc ||
        solver->work->Wnnz != first->work->Wnnz ||
        get_nnz(data->P) != first_Pnnz || get_nnz(data->A) != first_Annz ||
        get_nnz(data->G) != first_Gnnz) {
      return QOCO_DATA_VALIDATION_ERROR;
    }
  }
  return QOCO_NO_ERROR;
}

static QOCOCscMatrix* cuda_batch_construct_reference_kkt(QOCOSolver* solver)
{
  QOCOProblemData* data = solver->work->data;
  QOCOSettings* settings = solver->settings;
  QOCOCscMatrix* Kcsc = NULL;

  set_cpu_mode(1);
  Kcsc = construct_kkt(
      get_csc_matrix(data->P), get_csc_matrix(data->A), get_csc_matrix(data->G),
      get_csc_matrix(data->At), get_csc_matrix(data->Gt),
      settings->kkt_static_reg_A, data->n, data->m, data->p, data->l,
      data->nsoc, get_data_vectori(data->q), NULL, NULL, NULL, NULL, NULL,
      solver->work->Wnnz);
  set_cpu_mode(0);
  return Kcsc;
}

extern "C" QOCOInt qoco_cuda_batch_setup(QOCOBatchSolver* batch)
{
  if (!batch || !batch->solvers || batch->batch_count <= 0) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  if (!load_cuda_libraries()) {
    fprintf(stderr, "Failed to load CUDA libraries\n");
    return QOCO_SETUP_ERROR;
  }

  if (!cuda_batch_symbols_available()) {
    fprintf(stderr,
            "Loaded cuDSS does not provide cudssMatrixCreateBatchCsr and "
            "cudssMatrixCreateBatchDn\n");
    return QOCO_SETUP_ERROR;
  }

  QOCOInt validation = cuda_batch_validate_solvers(batch);
  if (validation != QOCO_NO_ERROR) {
    return qoco_error((enum qoco_error_code)validation);
  }

  qoco_cuda_batch_cleanup(batch);

  CudaBatchLinSysData* batch_data =
      (CudaBatchLinSysData*)qoco_calloc(1, sizeof(CudaBatchLinSysData));
  if (!batch_data) {
    return qoco_error(QOCO_MALLOC_ERROR);
  }

  QOCOSolver* first = batch->solvers[0];
  QOCOProblemData* first_problem = first->work->data;
  batch_data->batch_count = batch->batch_count;
  batch_data->Kn = first_problem->n + first_problem->m + first_problem->p;

  QOCOCscMatrix* Kcsc = cuda_batch_construct_reference_kkt(first);
  if (!Kcsc) {
    qoco_free(batch_data);
    return QOCO_SETUP_ERROR;
  }
  batch_data->nnz = Kcsc->nnz;

  QOCOFloat* unused_csr_val = NULL;
  csc_to_csr_device(Kcsc, &batch_data->d_csr_row_ptr,
                    &batch_data->d_csr_col_ind, &unused_csr_val, NULL, NULL,
                    NULL);
  cudaFree(unused_csr_val);
  free_qoco_csc_matrix(Kcsc);

  batch_data->h_csr_row_ptrs =
      (void**)qoco_malloc(batch_data->batch_count * sizeof(void*));
  batch_data->h_csr_col_inds =
      (void**)qoco_malloc(batch_data->batch_count * sizeof(void*));
  batch_data->h_csr_values =
      (void**)qoco_malloc(batch_data->batch_count * sizeof(void*));
  batch_data->h_rhs_values =
      (void**)qoco_malloc(batch_data->batch_count * sizeof(void*));
  batch_data->h_xyz_values =
      (void**)qoco_malloc(batch_data->batch_count * sizeof(void*));
  batch_data->h_nrows =
      (QOCOInt*)qoco_malloc(batch_data->batch_count * sizeof(QOCOInt));
  batch_data->h_ncols =
      (QOCOInt*)qoco_malloc(batch_data->batch_count * sizeof(QOCOInt));
  batch_data->h_nnz =
      (QOCOInt*)qoco_malloc(batch_data->batch_count * sizeof(QOCOInt));
  batch_data->h_nrhs =
      (QOCOInt*)qoco_malloc(batch_data->batch_count * sizeof(QOCOInt));
  batch_data->h_ld =
      (QOCOInt*)qoco_malloc(batch_data->batch_count * sizeof(QOCOInt));

  if (!batch_data->h_csr_row_ptrs || !batch_data->h_csr_col_inds ||
      !batch_data->h_csr_values || !batch_data->h_rhs_values ||
      !batch_data->h_xyz_values || !batch_data->h_nrows ||
      !batch_data->h_ncols || !batch_data->h_nnz || !batch_data->h_nrhs ||
      !batch_data->h_ld) {
    batch->batch_linsys_data = batch_data;
    qoco_cuda_batch_cleanup(batch);
    return qoco_error(QOCO_MALLOC_ERROR);
  }

  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    batch_data->h_csr_row_ptrs[item] = (void*)batch_data->d_csr_row_ptr;
    batch_data->h_csr_col_inds[item] = (void*)batch_data->d_csr_col_ind;
    batch_data->h_nrows[item] = batch_data->Kn;
    batch_data->h_ncols[item] = batch_data->Kn;
    batch_data->h_nnz[item] = batch_data->nnz;
    batch_data->h_nrhs[item] = 1;
    batch_data->h_ld[item] = batch_data->Kn;
  }

  CUDA_CHECK(cudaMalloc(&batch_data->d_csr_row_ptrs,
                        batch_data->batch_count * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&batch_data->d_csr_col_inds,
                        batch_data->batch_count * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&batch_data->d_csr_values,
                        batch_data->batch_count * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&batch_data->d_rhs_values,
                        batch_data->batch_count * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&batch_data->d_xyz_values,
                        batch_data->batch_count * sizeof(void*)));

  CUDA_CHECK(cudaMemcpy(batch_data->d_csr_row_ptrs,
                        batch_data->h_csr_row_ptrs,
                        batch_data->batch_count * sizeof(void*),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch_data->d_csr_col_inds,
                        batch_data->h_csr_col_inds,
                        batch_data->batch_count * sizeof(void*),
                        cudaMemcpyHostToDevice));

  cuda_batch_refresh_value_pointers(batch, batch_data);

  CUDSS_CHECK(g_cuda_funcs.cudssCreate(&batch_data->handle));
  CUDSS_CHECK(g_cuda_funcs.cudssConfigCreate(&batch_data->config));
  CUDSS_CHECK(
      g_cuda_funcs.cudssDataCreate(batch_data->handle, &batch_data->data));
  int value = 0;
  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(batch_data->config,
                                          CUDSS_CONFIG_USE_SUPERPANELS,
                                          (void*)&value, sizeof(int)));

  cudaDataType_t value_type =
      (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;

  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateBatchDn(
      &batch_data->rhs_batch, (int64_t)batch_data->batch_count,
      batch_data->h_nrows, batch_data->h_nrhs, batch_data->h_ld,
      batch_data->d_rhs_values, CUDA_R_32I, value_type,
      CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateBatchDn(
      &batch_data->xyz_batch, (int64_t)batch_data->batch_count,
      batch_data->h_nrows, batch_data->h_nrhs, batch_data->h_ld,
      batch_data->d_xyz_values, CUDA_R_32I, value_type,
      CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateBatchCsr(
      &batch_data->K_batch, (int64_t)batch_data->batch_count,
      batch_data->h_nrows, batch_data->h_ncols, batch_data->h_nnz,
      batch_data->d_csr_row_ptrs, NULL, batch_data->d_csr_col_inds,
      batch_data->d_csr_values, CUDA_R_32I, value_type,
      CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));

  CUDSS_CHECK(g_cuda_funcs.cudssExecute(
      batch_data->handle, CUDSS_PHASE_ANALYSIS, batch_data->config,
      batch_data->data, batch_data->K_batch, batch_data->xyz_batch,
      batch_data->rhs_batch));

  batch->batch_linsys_data = batch_data;
  batch->batch_linsys_stale = 0;
  return QOCO_NO_ERROR;
}

extern "C" void qoco_cuda_batch_cleanup(QOCOBatchSolver* batch)
{
  if (!batch || !batch->batch_linsys_data) {
    return;
  }

  CudaBatchLinSysData* batch_data =
      (CudaBatchLinSysData*)batch->batch_linsys_data;

  if (g_libs_loaded) {
    if (batch_data->K_batch) {
      g_cuda_funcs.cudssMatrixDestroy(batch_data->K_batch);
    }
    if (batch_data->rhs_batch) {
      g_cuda_funcs.cudssMatrixDestroy(batch_data->rhs_batch);
    }
    if (batch_data->xyz_batch) {
      g_cuda_funcs.cudssMatrixDestroy(batch_data->xyz_batch);
    }
    if (batch_data->data) {
      g_cuda_funcs.cudssDataDestroy(batch_data->handle, batch_data->data);
    }
    if (batch_data->config) {
      g_cuda_funcs.cudssConfigDestroy(batch_data->config);
    }
    if (batch_data->handle) {
      g_cuda_funcs.cudssDestroy(batch_data->handle);
    }
  }

  cudaFree(batch_data->d_csr_row_ptr);
  cudaFree(batch_data->d_csr_col_ind);
  cudaFree(batch_data->d_csr_row_ptrs);
  cudaFree(batch_data->d_csr_col_inds);
  cudaFree(batch_data->d_csr_values);
  cudaFree(batch_data->d_rhs_values);
  cudaFree(batch_data->d_xyz_values);

  qoco_free(batch_data->h_csr_row_ptrs);
  qoco_free(batch_data->h_csr_col_inds);
  qoco_free(batch_data->h_csr_values);
  qoco_free(batch_data->h_rhs_values);
  qoco_free(batch_data->h_xyz_values);
  qoco_free(batch_data->h_nrows);
  qoco_free(batch_data->h_ncols);
  qoco_free(batch_data->h_nnz);
  qoco_free(batch_data->h_nrhs);
  qoco_free(batch_data->h_ld);
  qoco_free(batch_data);

  batch->batch_linsys_data = NULL;
  batch->batch_linsys_stale = 1;
}

static void cuda_batch_set_matrix_values(CudaBatchLinSysData* batch_data)
{
  if (g_cuda_funcs.cudssMatrixSetBatchValues) {
    CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetBatchValues(
        batch_data->K_batch, batch_data->d_csr_values));
  }
}

static void cuda_batch_set_nt_identity(QOCOBatchSolver* batch,
                                       CudaBatchLinSysData* batch_data,
                                       const unsigned char* active)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }
    QOCOSolver* solver = batch->solvers[item];
    cudss_set_nt_identity((LinSysData*)solver->linsys_data,
                          solver->work->data->m);
  }
  cuda_batch_set_matrix_values(batch_data);
}

static void cuda_batch_update_nt(QOCOBatchSolver* batch,
                                 CudaBatchLinSysData* batch_data,
                                 const unsigned char* active)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }
    QOCOSolver* solver = batch->solvers[item];
    cudss_update_nt((LinSysData*)solver->linsys_data, solver->work->WtW,
                    solver->settings->kkt_static_reg_G,
                    solver->work->data->m);
  }
  cuda_batch_set_matrix_values(batch_data);
}

static void cuda_batch_factor(CudaBatchLinSysData* batch_data)
{
  qoco_cuda_timed_cudss_execute(
      batch_data->handle, CUDSS_PHASE_FACTORIZATION, batch_data->config,
      batch_data->data, batch_data->K_batch, batch_data->xyz_batch,
      batch_data->rhs_batch, 1);
}

static void cuda_batch_execute_solve(CudaBatchLinSysData* batch_data)
{
  qoco_cuda_timed_cudss_execute(
      batch_data->handle, CUDSS_PHASE_SOLVE, batch_data->config,
      batch_data->data, batch_data->K_batch, batch_data->xyz_batch,
      batch_data->rhs_batch, 1);
}

static void cuda_batch_pack_rhs(QOCOBatchSolver* batch,
                                CudaBatchLinSysData* batch_data,
                                const unsigned char* active,
                                unsigned char use_residual)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    LinSysData* linsys_data = (LinSysData*)batch->solvers[item]->linsys_data;
    if (active && !active[item]) {
      CUDA_CHECK(cudaMemset(linsys_data->d_rhs_matrix_data, 0,
                            batch_data->Kn * sizeof(QOCOFloat)));
    }
    else {
      QOCOFloat* rhs =
          use_residual ? linsys_data->d_xyz_matrix_data
                       : get_data_vectorf(batch->solvers[item]->work->rhs);
      CUDA_CHECK(cudaMemcpy(linsys_data->d_rhs_matrix_data, rhs,
                            batch_data->Kn * sizeof(QOCOFloat),
                            cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaMemset(linsys_data->d_xyz_matrix_data, 0,
                          batch_data->Kn * sizeof(QOCOFloat)));
  }
}

static void cuda_batch_copy_solve_output(QOCOBatchSolver* batch,
                                         CudaBatchLinSysData* batch_data,
                                         const unsigned char* active)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }
    LinSysData* linsys_data = (LinSysData*)batch->solvers[item]->linsys_data;
    QOCOFloat* x = get_data_vectorf(batch->solvers[item]->work->xyz);
    CUDA_CHECK(cudaMemcpy(x, linsys_data->d_xyz_matrix_data,
                          batch_data->Kn * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToDevice));
  }
}

static unsigned char cuda_batch_any_active(const unsigned char* active,
                                           QOCOInt n)
{
  for (QOCOInt i = 0; i < n; ++i) {
    if (active[i]) {
      return 1;
    }
  }
  return 0;
}

static void cuda_batch_solve_refined(QOCOBatchSolver* batch,
                                     CudaBatchLinSysData* batch_data,
                                     const unsigned char* active)
{
  cuda_batch_pack_rhs(batch, batch_data, active, 0);
  cuda_batch_execute_solve(batch_data);
  cuda_batch_copy_solve_output(batch, batch_data, active);

  unsigned char* refine_active =
      (unsigned char*)qoco_calloc(batch_data->batch_count, sizeof(unsigned char));
  QOCOInt* ir_used =
      (QOCOInt*)qoco_calloc(batch_data->batch_count, sizeof(QOCOInt));
  QOCOFloat* best_res =
      (QOCOFloat*)qoco_calloc(batch_data->batch_count, sizeof(QOCOFloat));
  if (!refine_active || !ir_used || !best_res) {
    qoco_free(refine_active);
    qoco_free(ir_used);
    qoco_free(best_res);
    return;
  }

  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }

    QOCOSolver* solver = batch->solvers[item];
    LinSysData* linsys_data = (LinSysData*)solver->linsys_data;
    QOCOWorkspace* work = solver->work;
    QOCOFloat* x = get_data_vectorf(work->xyz);
    QOCOFloat* b = get_data_vectorf(work->rhs);
    QOCOFloat* best_sol = get_data_vectorf(work->xyzbuff1);

    best_res[item] =
        compute_linsys_residual(linsys_data, work, b, x,
                                linsys_data->d_xyz_matrix_data);
    copy_arrayf(x, best_sol, batch_data->Kn);
    refine_active[item] =
        (best_res[item] >= solver->settings->ir_tol &&
         solver->settings->max_ir_iters > 0);
  }

  while (cuda_batch_any_active(refine_active, batch_data->batch_count)) {
    cuda_batch_pack_rhs(batch, batch_data, refine_active, 1);
    cuda_batch_execute_solve(batch_data);

    for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
      if (!refine_active[item]) {
        continue;
      }

      QOCOSolver* solver = batch->solvers[item];
      LinSysData* linsys_data = (LinSysData*)solver->linsys_data;
      QOCOWorkspace* work = solver->work;
      QOCOFloat* x = get_data_vectorf(work->xyz);
      QOCOFloat* b = get_data_vectorf(work->rhs);
      QOCOFloat* best_sol = get_data_vectorf(work->xyzbuff1);

      qoco_axpy(linsys_data->d_xyz_matrix_data, x, x, 1.0, batch_data->Kn);
      QOCOFloat new_res =
          compute_linsys_residual(linsys_data, work, b, x,
                                  linsys_data->d_xyz_matrix_data);

      if (new_res >= best_res[item]) {
        copy_arrayf(best_sol, x, batch_data->Kn);
        refine_active[item] = 0;
        continue;
      }

      ir_used[item]++;
      best_res[item] = new_res;
      copy_arrayf(x, best_sol, batch_data->Kn);
      refine_active[item] =
          (new_res >= solver->settings->ir_tol &&
           ir_used[item] < solver->settings->max_ir_iters);
    }
  }

  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (!active || active[item]) {
      batch->solvers[item]->work->ir_iters += ir_used[item];
    }
  }

  qoco_free(refine_active);
  qoco_free(ir_used);
  qoco_free(best_res);
}

static void cuda_batch_finish_item(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  stop_timer(&(work->solve_timer));

  unsigned char restored = 0;
  if (solver->sol->status == QOCO_NUMERICAL_ERROR) {
    restored = restore_best_iterate(solver);
  }
  unscale_variables(work);
  copy_solution(solver);
  if (solver->settings->verbose) {
    if (restored) {
      printf("Best iterate (%d) restored\n", work->best_iter);
    }
    print_footer(solver->sol,
                 (enum qoco_solve_status)solver->sol->status);
  }
}

static void cuda_batch_initialize_ipm(QOCOBatchSolver* batch,
                                      CudaBatchLinSysData* batch_data,
                                      unsigned char* active)
{
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    QOCOSolver* solver = batch->solvers[item];
    QOCOWorkspace* work = solver->work;
    QOCOProblemData* data = work->data;

    set_Wfull_identity(work->Wfull, work->Wnnzfull, work->Wsoc_idx, data);
    work->a = 1.0;

    QOCOFloat* rhs = get_data_vectorf(work->rhs);
    copy_and_negate_arrayf(get_data_vectorf(data->c), rhs, data->n);
    copy_arrayf(get_data_vectorf(data->b), &rhs[data->n], data->p);
    copy_arrayf(get_data_vectorf(data->h), &rhs[data->n + data->p], data->m);
  }

  cuda_batch_set_nt_identity(batch, batch_data, active);
  cuda_batch_factor(batch_data);
  cuda_batch_solve_refined(batch, batch_data, active);

  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    QOCOSolver* solver = batch->solvers[item];
    QOCOWorkspace* work = solver->work;
    QOCOProblemData* data = work->data;
    QOCOFloat* xyz = get_data_vectorf(work->xyz);

    copy_arrayf(xyz, get_data_vectorf(work->x), data->n);
    copy_arrayf(&xyz[data->n], get_data_vectorf(work->y), data->p);
    copy_arrayf(&xyz[data->n + data->p], get_data_vectorf(work->z), data->m);
    copy_and_negate_arrayf(&xyz[data->n + data->p], get_data_vectorf(work->s),
                           data->m);

    bring2cone(get_data_vectorf(work->s), get_data_vectori(work->soc_idx),
               data);
    bring2cone(get_data_vectorf(work->z), get_data_vectori(work->soc_idx),
               data);

    if (work->use_x0) {
      QOCOFloat* x0 = get_data_vectorf(work->x0);
      QOCOFloat* x = get_data_vectorf(work->x);
      QOCOFloat* Dinvruiz = get_data_vectorf(work->scaling->Dinvruiz);
      ew_product(x0, Dinvruiz, x, data->n);

      if (data->m > 0) {
        QOCOFloat* s = get_data_vectorf(work->s);
        QOCOFloat* h = get_data_vectorf(data->h);
        SpMv(data->G, x, s);
        qoco_axpy(s, h, s, -1.0, data->m);
        bring2cone(s, get_data_vectori(work->soc_idx), data);
      }
    }
  }
}

static void cuda_batch_after_affine_solve(QOCOBatchSolver* batch,
                                          CudaBatchLinSysData* batch_data,
                                          const unsigned char* active)
{
  (void)batch_data;
  for (QOCOInt item = 0; item < batch->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }
    QOCOSolver* solver = batch->solvers[item];
    QOCOWorkspace* work = solver->work;
    QOCOProblemData* data = work->data;

    QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
    QOCOInt* Wsoc_idx = get_data_vectori(work->Wsoc_idx);
    QOCOInt* soc_idx = get_data_vectori(work->soc_idx);
    QOCOFloat* lambda = get_data_vectorf(work->lambda);
    QOCOFloat* Ds = get_data_vectorf(work->Ds);
    QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
    QOCOInt* q = get_data_vectori(data->q);
    QOCOFloat* xyz = get_data_vectorf(work->xyz);
    QOCOFloat* Dzaff = &xyz[data->n + data->p];

    nt_multiply(Wfull, Wsoc_idx, soc_idx, Dzaff, ubuff1, data->l, data->m,
                data->nsoc, q);
    copy_and_negate_arrayf(ubuff1, ubuff1, data->m);
    qoco_axpy(lambda, ubuff1, ubuff1, -1.0, data->m);
    nt_multiply(Wfull, Wsoc_idx, soc_idx, ubuff1, Ds, data->l, data->m,
                data->nsoc, q);

    compute_centering(solver);
    construct_kkt_comb_rhs(work);
  }
}

static void cuda_batch_after_combined_solve(QOCOBatchSolver* batch,
                                            CudaBatchLinSysData* batch_data,
                                            const unsigned char* active,
                                            QOCOInt iter)
{
  (void)batch_data;
  for (QOCOInt item = 0; item < batch->batch_count; ++item) {
    if (active && !active[item]) {
      continue;
    }

    QOCOSolver* solver = batch->solvers[item];
    QOCOWorkspace* work = solver->work;
    QOCOProblemData* data = work->data;

    if (check_nan(work->xyz)) {
      work->a = 0.0;
      solver->sol->iters = iter;
      solver->sol->ir_iters += work->ir_iters;
      continue;
    }

    QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
    QOCOInt* Wsoc_idx = get_data_vectori(work->Wsoc_idx);
    QOCOInt* soc_idx = get_data_vectori(work->soc_idx);
    QOCOFloat* lambda = get_data_vectorf(work->lambda);
    QOCOFloat* Ds = get_data_vectorf(work->Ds);
    QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
    QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
    QOCOFloat* ubuff3 = get_data_vectorf(work->ubuff3);
    QOCOInt* q = get_data_vectori(data->q);
    QOCOFloat* xyz = get_data_vectorf(work->xyz);
    QOCOFloat* Dz = &xyz[data->n + data->p];

    cone_division(lambda, Ds, ubuff1, data->l, data->nsoc, q, soc_idx);
    nt_multiply(Wfull, Wsoc_idx, soc_idx, Dz, ubuff2, data->l, data->m,
                data->nsoc, q);
    qoco_axpy(ubuff2, ubuff1, ubuff3, -1.0, data->m);
    nt_multiply(Wfull, Wsoc_idx, soc_idx, ubuff3, Ds, data->l, data->m,
                data->nsoc, q);

    QOCOFloat a =
        qoco_min(linesearch(get_data_vectorf(work->s), Ds, 0.99, solver),
                 linesearch(get_data_vectorf(work->z), Dz, 0.99, solver));
    work->a = a;

    QOCOFloat* Dx = xyz;
    QOCOFloat* Dy = &xyz[data->n];
    qoco_axpy(Dx, get_data_vectorf(work->x), get_data_vectorf(work->x), a,
              data->n);
    qoco_axpy(Ds, get_data_vectorf(work->s), get_data_vectorf(work->s), a,
              data->m);
    qoco_axpy(Dy, get_data_vectorf(work->y), get_data_vectorf(work->y), a,
              data->p);
    qoco_axpy(Dz, get_data_vectorf(work->z), get_data_vectorf(work->z), a,
              data->m);

    solver->sol->iters = iter;
    solver->sol->ir_iters += work->ir_iters;
    if (solver->settings->verbose) {
      log_iter(solver);
    }
  }
}

extern "C" QOCOInt qoco_cuda_batch_solve(QOCOBatchSolver* batch)
{
  if (!batch || !batch->solvers || !batch->statuses) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  if (!batch->batch_linsys_data || batch->batch_linsys_stale) {
    QOCOInt exit = qoco_cuda_batch_setup(batch);
    if (exit != QOCO_NO_ERROR) {
      return exit;
    }
  }

  CudaBatchLinSysData* batch_data =
      (CudaBatchLinSysData*)batch->batch_linsys_data;
  cuda_batch_refresh_value_pointers(batch, batch_data);

  unsigned char* active =
      (unsigned char*)qoco_calloc(batch_data->batch_count, sizeof(unsigned char));
  if (!active) {
    return qoco_error(QOCO_MALLOC_ERROR);
  }

  QOCOInt max_iters = 0;
  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    QOCOSolver* solver = batch->solvers[item];
    if (qoco_validate_settings(solver->settings)) {
      qoco_free(active);
      return qoco_error(QOCO_SETTINGS_VALIDATION_ERROR);
    }
    active[item] = 1;
    batch->statuses[item] = QOCO_UNSOLVED;
    solver->sol->status = QOCO_UNSOLVED;
    max_iters = qoco_max(max_iters, solver->settings->max_iters);
    start_timer(&(solver->work->solve_timer));
    if (solver->settings->verbose) {
      print_header(solver);
    }
  }

  log_ipm_iter(0);
  cuda_batch_initialize_ipm(batch, batch_data, active);

  for (QOCOInt iter = 1; iter <= max_iters; ++iter) {
    for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
      if (!active[item]) {
        continue;
      }

      QOCOSolver* solver = batch->solvers[item];
      QOCOWorkspace* work = solver->work;
      QOCOProblemData* data = work->data;

      compute_kkt_residual(data, work->x, work->y, work->s, work->z,
                           work->kktres, solver->settings->kkt_static_reg_P,
                           work->xyzbuff1, work->xbuff, work->ubuff1);
      solver->sol->obj = compute_objective(
          data, work->x, work->xbuff, solver->settings->kkt_static_reg_P,
          work->scaling->k);
      work->mu = compute_mu(work->s, work->z, data->m);

      if (check_stopping(solver)) {
        cuda_batch_finish_item(solver);
        batch->statuses[item] = solver->sol->status;
        active[item] = 0;
      }
    }

    if (!cuda_batch_any_active(active, batch_data->batch_count)) {
      qoco_free(active);
      return QOCO_NO_ERROR;
    }

    log_ipm_iter(iter);

    for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
      if (!active[item]) {
        continue;
      }
      QOCOSolver* solver = batch->solvers[item];
      compute_nt_scaling(solver->work);
      solver->work->ir_iters = 0;
    }

    cuda_batch_update_nt(batch, batch_data, active);
    cuda_batch_factor(batch_data);

    for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
      if (active[item]) {
        construct_kkt_aff_rhs(batch->solvers[item]->work);
      }
    }
    cuda_batch_solve_refined(batch, batch_data, active);

    cuda_batch_after_affine_solve(batch, batch_data, active);
    cuda_batch_solve_refined(batch, batch_data, active);
    cuda_batch_after_combined_solve(batch, batch_data, active, iter);
  }

  for (QOCOInt item = 0; item < batch_data->batch_count; ++item) {
    if (!active[item]) {
      continue;
    }
    QOCOSolver* solver = batch->solvers[item];
    solver->sol->status = QOCO_MAX_ITER;
    restore_best_iterate(solver);
    cuda_batch_finish_item(solver);
    batch->statuses[item] = solver->sol->status;
  }

  qoco_free(active);
  return QOCO_NO_ERROR;
}

static const char* cudss_name() { return "cuda/cuDSS"; }

LinSysBackend backend = {.linsys_name = cudss_name,
                         .linsys_setup = cudss_setup,
                         .linsys_set_nt_identity = cudss_set_nt_identity,
                         .linsys_update_nt = cudss_update_nt,
                         .linsys_update_data = cudss_update_data,
                         .linsys_factor = cudss_factor,
                         .linsys_solve = cudss_solve,
                         .linsys_cleanup = cudss_cleanup};
