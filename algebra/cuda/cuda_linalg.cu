/**
 * @file cuda_linalg.cu
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "cuda_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "common_linalg.h"
#include "cudss_backend.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// CUDA kernels
__global__ void copy_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y, QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

__global__ void copy_and_negate_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y,
                                              QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = -x[idx];
  }
}

__global__ void copy_arrayi_kernel(const QOCOInt* x, QOCOInt* y, QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

__global__ void scale_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y,
                                    QOCOFloat s, QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = s * x[idx];
  }
}

__global__ void axpy_kernel(const QOCOFloat* x, const QOCOFloat* y,
                            QOCOFloat* z, QOCOFloat a, QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = a * x[idx] + y[idx];
  }
}

// Construct A on CPU and set device pointer to NULL.
QOCOMatrix* new_qoco_matrix(const QOCOCscMatrix* A)
{
  QOCOMatrix* M = (QOCOMatrix*)qoco_malloc(sizeof(QOCOMatrix));
  M->d_csc = NULL;  // Initialize device matrix to NULL

  if (A) {
    QOCOInt m = A->m;
    QOCOInt n = A->n;
    QOCOInt nnz = A->nnz;

    // Allocate host CSC matrix
    QOCOCscMatrix* Mcsc = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));
    QOCOFloat* x = (QOCOFloat*)qoco_malloc(nnz * sizeof(QOCOFloat));
    QOCOInt* p = (QOCOInt*)qoco_malloc((n + 1) * sizeof(QOCOInt));
    QOCOInt* i = (QOCOInt*)qoco_malloc(nnz * sizeof(QOCOInt));

    copy_arrayf(A->x, x, nnz);
    copy_arrayi(A->i, i, nnz);
    copy_arrayi(A->p, p, n + 1);

    Mcsc->m = m;
    Mcsc->n = n;
    Mcsc->nnz = nnz;
    Mcsc->x = x;
    Mcsc->i = i;
    Mcsc->p = p;
    M->csc = Mcsc;

    // Allocate device CSC matrix structure and data
    QOCOCscMatrix* d_Mcsc;
    QOCOFloat* d_x;
    QOCOInt* d_p;
    QOCOInt* d_i;
    
    // Allocate structure on device
    CUDA_CHECK(cudaMalloc(&d_Mcsc, sizeof(QOCOCscMatrix)));
    CUDA_CHECK(cudaMalloc(&d_x, nnz * sizeof(QOCOFloat)));
    CUDA_CHECK(cudaMalloc(&d_p, (n + 1) * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMalloc(&d_i, nnz * sizeof(QOCOInt)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, x, nnz * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, p, (n + 1) * sizeof(QOCOInt), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_i, i, nnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
    
    // Create structure on host with device pointers, then copy to device
    QOCOCscMatrix h_d_Mcsc;
    h_d_Mcsc.m = m;
    h_d_Mcsc.n = n;
    h_d_Mcsc.nnz = nnz;
    h_d_Mcsc.x = d_x;
    h_d_Mcsc.i = d_i;
    h_d_Mcsc.p = d_p;
    
    // Copy structure to device
    CUDA_CHECK(cudaMemcpy(d_Mcsc, &h_d_Mcsc, sizeof(QOCOCscMatrix), cudaMemcpyHostToDevice));
    
    // Store device pointer for kernel calls
    M->d_csc = d_Mcsc;
  }
  else {
    QOCOCscMatrix* Mcsc = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));
    Mcsc->m = 0;
    Mcsc->n = 0;
    Mcsc->nnz = 0;
    Mcsc->x = NULL;
    Mcsc->i = NULL;
    Mcsc->p = NULL;
    M->csc = Mcsc;
    M->d_csc = NULL;
  }

  return M;
}

// Construct x on CPU copy vector to GPU.
QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n)
{
  QOCOVectorf* v = (QOCOVectorf*)qoco_malloc(sizeof(QOCOVectorf));
  QOCOFloat* vdata = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * n);

  if (x) {
    copy_arrayf(x, vdata, n);
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      vdata[i] = 0.0;
    }
  }

  QOCOFloat* d_vdata;
  CUDA_CHECK(cudaMalloc(&d_vdata, sizeof(QOCOFloat) * n));
  CUDA_CHECK(cudaMemcpy(d_vdata, vdata, sizeof(QOCOFloat) * n,
                        cudaMemcpyHostToDevice));

  v->len = n;
  v->data = vdata;
  v->d_data = d_vdata;

  return v;
}

void free_qoco_matrix(QOCOMatrix* A)
{
  if (A) {
    // Free host CSC matrix
    free_qoco_csc_matrix(A->csc);
    
    // Free device CSC matrix if allocated
    if (A->d_csc) {
      // Copy structure from device to get data pointers for cleanup
      QOCOCscMatrix d_csc_host;
      CUDA_CHECK(cudaMemcpy(&d_csc_host, A->d_csc, sizeof(QOCOCscMatrix), cudaMemcpyDeviceToHost));
      
      // Free device data arrays
      if (d_csc_host.x) cudaFree(d_csc_host.x);
      if (d_csc_host.i) cudaFree(d_csc_host.i);
      if (d_csc_host.p) cudaFree(d_csc_host.p);
      
      // Free device structure
      cudaFree(A->d_csc);
    }
    
    qoco_free(A);
  }
}

void free_qoco_vectorf(QOCOVectorf* x)
{
  if (x) {
    if (x->data)
      qoco_free(x->data);
    if (x->d_data)
      cudaFree(x->d_data);
    qoco_free(x);
  }
}

QOCOInt get_nnz(const QOCOMatrix* A) { return A->csc->nnz; }

QOCOFloat get_element_vectorf(const QOCOVectorf* x, QOCOInt idx)
{
  return x->data[idx];
}

QOCOFloat* get_pointer_vectorf(const QOCOVectorf* x, QOCOInt idx)
{
  return &x->data[idx];
}

// Static flag to track when we're in compute_scaling_statistics
static int in_scaling_statistics_mode = 0;

void set_scaling_statistics_mode(int active)
{
  in_scaling_statistics_mode = active;
}

QOCOFloat* get_data_vectorf(const QOCOVectorf* x)
{
  // During compute_scaling_statistics, return host pointer for CPU access
  if (in_scaling_statistics_mode) {
    return x->data;
  }
  // During equilibration/setup (CPU phase), return host pointer
  // During solve (GPU phase), return device pointer to avoid CPU-GPU copies
  return x->d_data;
}

QOCOInt get_length_vectorf(const QOCOVectorf* x) { return x->len; }

void sync_vector_to_host(QOCOVectorf* v)
{
  if (v && v->data && v->d_data) {
    CUDA_CHECK(cudaMemcpy(v->data, v->d_data, v->len * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToHost));
  }
}

void sync_vector_to_device(const QOCOVectorf* v)
{
  if (v && v->data && v->d_data) {
    CUDA_CHECK(cudaMemcpy(v->d_data, v->data, v->len * sizeof(QOCOFloat),
                          cudaMemcpyHostToDevice));
  }
}

QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M) { return M->csc; }

void col_inf_norm_USymm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during equilibration on CPU
  col_inf_norm_USymm(get_csc_matrix(M), norm);
}

void col_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during equilibration on CPU
  col_inf_norm(get_csc_matrix(M), norm);
}

void row_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during equilibration on CPU
  row_inf_norm(get_csc_matrix(M), norm);
}

void row_col_scale_matrix(QOCOMatrix* M, const QOCOFloat* E, const QOCOFloat* D)
{
  // Called during equilibration on CPU
  row_col_scale(get_csc_matrix(M), (QOCOFloat*)E, (QOCOFloat*)D);
}

void set_element_vectorf(QOCOVectorf* x, QOCOInt idx, QOCOFloat data)
{
  // Called during equilibration on CPU
  x->data[idx] = data;
}

void reciprocal_vectorf(const QOCOVectorf* input, QOCOVectorf* output)
{
  // Called during equilibration on CPU
  for (QOCOInt i = 0; i < input->len; ++i) {
    output->data[i] = safe_div(1.0, input->data[i]);
  }
}

QOCOCscMatrix* new_qoco_csc_matrix(const QOCOCscMatrix* A)
{
  QOCOCscMatrix* M = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));

  if (A) {
    QOCOInt m = A->m;
    QOCOInt n = A->n;
    QOCOInt nnz = A->nnz;

    QOCOFloat* x = (QOCOFloat*)qoco_malloc(nnz * sizeof(QOCOFloat));
    QOCOInt* p = (QOCOInt*)qoco_malloc((n + 1) * sizeof(QOCOInt));
    QOCOInt* i = (QOCOInt*)qoco_malloc(nnz * sizeof(QOCOInt));

    copy_arrayf(A->x, x, nnz);
    copy_arrayi(A->i, i, nnz);
    copy_arrayi(A->p, p, n + 1);

    M->m = m;
    M->n = n;
    M->nnz = nnz;
    M->x = x;
    M->i = i;
    M->p = p;
  }
  else {
    M->m = 0;
    M->n = 0;
    M->nnz = 0;
    M->x = NULL;
    M->i = NULL;
    M->p = NULL;
  }

  return M;
}

void free_qoco_csc_matrix(QOCOCscMatrix* A)
{
  if (A) {
    free(A->x);
    free(A->i);
    free(A->p);
    free(A);
  }
}

void copy_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  // Check if pointers are on device - handle errors gracefully
  cudaPointerAttributes attrs_x, attrs_y;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  cudaError_t err_y = cudaPointerGetAttributes(&attrs_y, y);

  // Check if we have mixed host/device pointers
  int x_is_device = (err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice);
  int y_is_device = (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice);

  if (x_is_device && y_is_device) {
    // Both on device - use CUDA kernel
    copy_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y, n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else if (x_is_device && !y_is_device) {
    // x on device, y on host - use cudaMemcpy device to host
    CUDA_CHECK(cudaMemcpy(y, x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
  }
  else if (!x_is_device && y_is_device) {
    // x on host, y on device - use cudaMemcpy host to device
    CUDA_CHECK(cudaMemcpy(y, x, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  }
  else {
    // Both on host - use CPU
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = x[i];
    }
  }
}

void copy_and_negate_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  // Check if pointers are on device - handle errors gracefully
  cudaPointerAttributes attrs_x, attrs_y;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  cudaError_t err_y = cudaPointerGetAttributes(&attrs_y, y);

  // Check if we have mixed host/device pointers
  int x_is_device = (err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice);
  int y_is_device = (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice);

  if (x_is_device && y_is_device) {
    // Both on device - use CUDA kernel
    copy_and_negate_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y,
                                                            n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else if (x_is_device && !y_is_device) {
    // x on device, y on host - copy to temp buffer, negate on host
    QOCOFloat* temp = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
    CUDA_CHECK(cudaMemcpy(temp, x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = -temp[i];
    }
    qoco_free(temp);
  }
  else if (!x_is_device && y_is_device) {
    // x on host, y on device - negate on host, then copy to device
    QOCOFloat* temp = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
    for (QOCOInt i = 0; i < n; ++i) {
      temp[i] = -x[i];
    }
    CUDA_CHECK(cudaMemcpy(y, temp, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
    qoco_free(temp);
  }
  else {
    // Both on host - use CPU
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = -x[i];
    }
  }
}

// Only called by CPU during setup.
void copy_arrayi(const QOCOInt* x, QOCOInt* y, QOCOInt n)
{
  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

// TODO: Don't create and destroy cublas handle for each dot product. One way around this is to create a custom kernel for the dot product. 
QOCOFloat qoco_dot(const QOCOFloat* u, const QOCOFloat* v, QOCOInt n)
{
  qoco_assert(u || n == 0);
  qoco_assert(v || n == 0);

  if (n == 0)
    return 0.0;

  // Check if pointers are on device
  cudaPointerAttributes attrs_u, attrs_v;
  cudaError_t err_u = cudaPointerGetAttributes(&attrs_u, u);
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  int u_is_device = (err_u == cudaSuccess && attrs_u.type == cudaMemoryTypeDevice);
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);

  // If any are device, copy both to host and compute
  if (u_is_device || v_is_device) {
    QOCOFloat* u_host = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
    QOCOFloat* v_host = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
    CUDA_CHECK(cudaMemcpy(u_host, u, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_host, v, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));

    QOCOFloat x = 0.0;
    for (QOCOInt i = 0; i < n; ++i) {
      x += u_host[i] * v_host[i];
    }
    qoco_free(u_host);
    qoco_free(v_host);
    return x;
  } else {
    QOCOFloat x = 0.0;
    for (QOCOInt i = 0; i < n; ++i) {
      x += u[i] * v[i];
    }
    return x;
  }
}

QOCOInt max_arrayi(const QOCOInt* x, QOCOInt n)
{
  QOCOInt max = -QOCOInt_MAX;
  for (QOCOInt i = 0; i < n; ++i) {
    max = qoco_max(max, x[i]);
  }
  return max;
}

void scale_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  if (n == 0)
    return;

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  // Check if pointers are on device - handle errors gracefully
  cudaPointerAttributes attrs_x, attrs_y;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  cudaError_t err_y = cudaPointerGetAttributes(&attrs_y, y);

  // If either pointer check succeeds and one is on device, use CUDA
  if ((err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice) ||
      (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice)) {
    if (err_x == cudaSuccess && err_y == cudaSuccess &&
        attrs_x.type == cudaMemoryTypeDevice &&
        attrs_y.type == cudaMemoryTypeDevice) {
      scale_arrayf_kernel<<<numBlocks, blockSize>>>(x, y, s, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = s * x[i];
    }
  }
}

void qoco_axpy(const QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z,
               QOCOFloat a, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  if (n == 0)
    return;

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  // Check if pointers are on device - handle errors gracefully
  cudaPointerAttributes attrs_x, attrs_y, attrs_z;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  cudaError_t err_y = cudaPointerGetAttributes(&attrs_y, y);
  cudaError_t err_z = cudaPointerGetAttributes(&attrs_z, z);

  // If any pointer check succeeds and indicates device memory, use CUDA
  if ((err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice) ||
      (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice) ||
      (err_z == cudaSuccess && attrs_z.type == cudaMemoryTypeDevice)) {
    if (err_x == cudaSuccess && err_y == cudaSuccess && err_z == cudaSuccess &&
        attrs_x.type == cudaMemoryTypeDevice &&
        attrs_y.type == cudaMemoryTypeDevice &&
        attrs_z.type == cudaMemoryTypeDevice) {
      axpy_kernel<<<numBlocks, blockSize>>>(x, y, z, a, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      z[i] = a * x[i] + y[i];
    }
  }
}

__global__ void ew_product_kernel(const QOCOFloat* x, const QOCOFloat* y,
                                  QOCOFloat* z, QOCOInt n)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = x[idx] * y[idx];
  }
}

void ew_product(QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);
  qoco_assert(z || n == 0);

  if (n == 0)
    return;

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  cudaPointerAttributes attrs_x, attrs_y, attrs_z;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  cudaError_t err_y = cudaPointerGetAttributes(&attrs_y, y);
  cudaError_t err_z = cudaPointerGetAttributes(&attrs_z, z);

  int x_is_device = (err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice);
  int y_is_device = (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice);
  int z_is_device = (err_z == cudaSuccess && attrs_z.type == cudaMemoryTypeDevice);

  if ((x_is_device || y_is_device || z_is_device) &&
      !(x_is_device && y_is_device && z_is_device)) {
    fprintf(stderr,
            "Error in ew_product: mixed memory spaces "
            "(x_is_device=%d, y_is_device=%d, z_is_device=%d). "
            "x=%p, y=%p, z=%p\n",
            x_is_device, y_is_device, z_is_device, (void*)x, (void*)y, (void*)z);
    exit(1);
  }

  if (x_is_device && y_is_device && z_is_device) {
    ew_product_kernel<<<numBlocks, blockSize>>>(x, y, z, n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      z[i] = x[i] * y[i];
    }
  }
}

// CUDA kernel for device-side USpMv
// Each thread handles one column of the matrix
__global__ void USpMv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  QOCOInt i = blockIdx.x;
  if (i >= M->n) return;
  
  // Process all nonzeros in column i
  for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
    int row = M->i[j];
    QOCOFloat val = M->x[j] * v[i];
    
    // Add to r[row] using atomic (multiple columns can write to same row)
    atomicAdd(&r[row], val);
    
    // If off-diagonal, also add to r[i] (symmetric part)
    // Note: r[i] is only written by thread i, but we use atomic for safety
    if (row != i) {
      atomicAdd(&r[i], M->x[j] * v[row]);
    }
  }
}

// CUDA kernel for device-side SpMv
// Each thread handles one column of the matrix
__global__ void SpMv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  QOCOInt col = blockIdx.x;
  if (col >= M->n) return;
  
  // Process all nonzeros in column col
  for (QOCOInt idx = M->p[col]; idx < M->p[col + 1]; idx++) {
    QOCOInt row = M->i[idx];
    QOCOFloat val = M->x[idx] * v[col];
    
    // Add to r[row] using atomic (multiple columns can write to same row)
    atomicAdd(&r[row], val);
  }
}

// CUDA kernel for device-side SpMtv
// Each thread handles one column of the result
__global__ void SpMtv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  QOCOInt col = blockIdx.x;
  if (col >= M->n) return;
  
  QOCOFloat sum = 0.0;
  
  // Process all nonzeros in column col
  for (QOCOInt idx = M->p[col]; idx < M->p[col + 1]; idx++) {
    QOCOInt row = M->i[idx];
    sum += M->x[idx] * v[row];
  }
  
  r[col] = sum;
}

// Host function that handles both host and device pointers
// Requires that matrix and vectors are all on the same memory space (all CPU or all GPU)
void USpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // Check if vector pointers are on device first
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // Determine if vectors are on device
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Check if vectors are on different memory spaces
  if (v_is_device != r_is_device) {
    fprintf(stderr, "Error in USpMv: Input and output vectors are on different memory spaces\n");
    exit(1);
  }

  // If vectors are on device, assume M is also on device (structure pointer)
  // and we need to copy metadata from device first
  if (v_is_device && r_is_device) {
    // M is a device pointer - copy metadata to host
    QOCOCscMatrix M_host;
    CUDA_CHECK(cudaMemcpy(&M_host, M, sizeof(QOCOCscMatrix), cudaMemcpyDeviceToHost));
    
    // All on device - use GPU kernel
    // First initialize result vector to zero
    CUDA_CHECK(cudaMemset(r, 0, M_host.n * sizeof(QOCOFloat)));
    // Launch kernel with one thread per column
    USpMv_kernel<<<M_host.n, 1>>>(M, v, r);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    // Vectors are on host - check if matrix data is on device
    cudaPointerAttributes attrs_Mx, attrs_Mp, attrs_Mi;
    cudaError_t err_Mx = cudaPointerGetAttributes(&attrs_Mx, M->x);
    cudaError_t err_Mp = cudaPointerGetAttributes(&attrs_Mp, M->p);
    cudaError_t err_Mi = cudaPointerGetAttributes(&attrs_Mi, M->i);

    // Determine if matrix data is on device
    int Mx_is_device = (err_Mx == cudaSuccess && attrs_Mx.type == cudaMemoryTypeDevice);
    int Mp_is_device = (err_Mp == cudaSuccess && attrs_Mp.type == cudaMemoryTypeDevice);
    int Mi_is_device = (err_Mi == cudaSuccess && attrs_Mi.type == cudaMemoryTypeDevice);
    int M_is_device = Mx_is_device || Mp_is_device || Mi_is_device;

    // Check for mixed memory spaces - raise error if mixed
    if (M_is_device) {
      fprintf(stderr, "Error in USpMv: Matrix is on device but vectors are on host\n");
      exit(1);
    }

    // All on host - use CPU implementation
    for (QOCOInt i = 0; i < M->n; i++) {
      r[i] = 0.0;
      for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
        int row = M->i[j];
        r[row] += M->x[j] * v[i];
        if (row != i)
          r[i] += M->x[j] * v[row];
      }
    }
  }
}

void SpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // Check if vector pointers are on device first
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // Determine if vectors are on device
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Check if vectors are on different memory spaces
  if (v_is_device != r_is_device) {
    fprintf(stderr, "Error in SpMv: Input and output vectors are on different memory spaces\n");
    exit(1);
  }

  // If vectors are on device, assume M is also on device (structure pointer)
  // and we need to copy metadata from device first
  if (v_is_device && r_is_device) {
    // M is a device pointer - copy metadata to host
    QOCOCscMatrix M_host;
    CUDA_CHECK(cudaMemcpy(&M_host, M, sizeof(QOCOCscMatrix), cudaMemcpyDeviceToHost));
    
    // All on device - use GPU kernel
    // First initialize result vector to zero
    CUDA_CHECK(cudaMemset(r, 0, M_host.m * sizeof(QOCOFloat)));
    // Launch kernel with one thread per column
    SpMv_kernel<<<M_host.n, 1>>>(M, v, r);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    // Vectors are on host - check if matrix data is on device
    cudaPointerAttributes attrs_Mx, attrs_Mp, attrs_Mi;
    cudaError_t err_Mx = cudaPointerGetAttributes(&attrs_Mx, M->x);
    cudaError_t err_Mp = cudaPointerGetAttributes(&attrs_Mp, M->p);
    cudaError_t err_Mi = cudaPointerGetAttributes(&attrs_Mi, M->i);

    // Determine if matrix data is on device
    int Mx_is_device = (err_Mx == cudaSuccess && attrs_Mx.type == cudaMemoryTypeDevice);
    int Mp_is_device = (err_Mp == cudaSuccess && attrs_Mp.type == cudaMemoryTypeDevice);
    int Mi_is_device = (err_Mi == cudaSuccess && attrs_Mi.type == cudaMemoryTypeDevice);
    int M_is_device = Mx_is_device || Mp_is_device || Mi_is_device;

    // Check for mixed memory spaces - raise error if mixed
    if (M_is_device) {
      fprintf(stderr, "Error in SpMv: Matrix is on device but vectors are on host\n");
      exit(1);
    }

    // All on host - use CPU implementation
    for (QOCOInt i = 0; i < M->m; i++) {
      r[i] = 0.0;
    }
    for (QOCOInt col = 0; col < M->n; col++) {
      for (QOCOInt idx = M->p[col]; idx < M->p[col + 1]; idx++) {
        QOCOInt row = M->i[idx];
        r[row] += M->x[idx] * v[col];
      }
    }
  }
}

void SpMtv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // Check if vector pointers are on device first
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // Determine if vectors are on device
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Check if vectors are on different memory spaces
  if (v_is_device != r_is_device) {
    fprintf(stderr, "Error in SpMtv: Input and output vectors are on different memory spaces\n");
    exit(1);
  }

  // If vectors are on device, assume M is also on device (structure pointer)
  // and we need to copy metadata from device first
  if (v_is_device && r_is_device) {
    // M is a device pointer - copy metadata to host
    QOCOCscMatrix M_host;
    CUDA_CHECK(cudaMemcpy(&M_host, M, sizeof(QOCOCscMatrix), cudaMemcpyDeviceToHost));
    
    // All on device - use GPU kernel
    // First initialize result vector to zero
    CUDA_CHECK(cudaMemset(r, 0, M_host.n * sizeof(QOCOFloat)));
    // Launch kernel with one thread per column
    SpMtv_kernel<<<M_host.n, 1>>>(M, v, r);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    // Vectors are on host - check if matrix data is on device
    cudaPointerAttributes attrs_Mx, attrs_Mp, attrs_Mi;
    cudaError_t err_Mx = cudaPointerGetAttributes(&attrs_Mx, M->x);
    cudaError_t err_Mp = cudaPointerGetAttributes(&attrs_Mp, M->p);
    cudaError_t err_Mi = cudaPointerGetAttributes(&attrs_Mi, M->i);

    // Determine if matrix data is on device
    int Mx_is_device = (err_Mx == cudaSuccess && attrs_Mx.type == cudaMemoryTypeDevice);
    int Mp_is_device = (err_Mp == cudaSuccess && attrs_Mp.type == cudaMemoryTypeDevice);
    int Mi_is_device = (err_Mi == cudaSuccess && attrs_Mi.type == cudaMemoryTypeDevice);
    int M_is_device = Mx_is_device || Mp_is_device || Mi_is_device;

    // Check for mixed memory spaces - raise error if mixed
    if (M_is_device) {
      fprintf(stderr, "Error in SpMtv: Matrix is on device but vectors are on host\n");
      exit(1);
    }

    // All on host - use CPU implementation
    for (QOCOInt i = 0; i < M->n; ++i) {
      r[i] = 0.0;
    }

    for (QOCOInt i = 0; i < M->n; i++) {
      for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
        r[i] += M->x[j] * v[M->i[j]];
      }
    }
  }
}

void USpMv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  // Check if vectors are on device
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // If cudaPointerGetAttributes fails, assume host pointer
  // (This can happen for host pointers not managed by CUDA)
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Use device matrix only if BOTH vectors are on device and device matrix exists
  if (v_is_device && r_is_device) {
    if (!M->d_csc) {
      fprintf(stderr, "Error in USpMv_matrix: Vectors are on device but device matrix (d_csc) is NULL. "
                      "Matrix was likely created without device allocation. "
                      "This can happen if the matrix was created with A=NULL or if device allocation failed.\n");
      exit(1);
    }
    // Use device matrix structure - pass device pointer to structure
    USpMv(M->d_csc, v, r);
  }
  else if (!v_is_device && !r_is_device) {
    // Both vectors on host, use host matrix
    USpMv(M->csc, v, r);
  }
  else {
    // Mixed: one vector on device, one on host - this is an error
    fprintf(stderr, "Error in USpMv_matrix: Input and output vectors are on different memory spaces. "
                    "v_is_device=%d, r_is_device=%d\n", v_is_device, r_is_device);
    exit(1);
  }
}

void SpMv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  // Check if vectors are on device
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // If cudaPointerGetAttributes fails, assume host pointer
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Use device matrix only if BOTH vectors are on device and device matrix exists
  if (v_is_device && r_is_device) {
    if (!M->d_csc) {
      fprintf(stderr, "Error in SpMv_matrix: Vectors are on device but device matrix (d_csc) is NULL. "
                      "Matrix was likely created without device allocation. "
                      "This can happen if the matrix was created with A=NULL or if device allocation failed.\n");
      exit(1);
    }
    // Use device matrix structure - pass device pointer to structure
    SpMv(M->d_csc, v, r);
  }
  else if (!v_is_device && !r_is_device) {
    // Both vectors on host, use host matrix
    SpMv(M->csc, v, r);
  }
  else {
    // Mixed: one vector on device, one on host - this is an error
    fprintf(stderr, "Error in SpMv_matrix: Input and output vectors are on different memory spaces. "
                    "v_is_device=%d, r_is_device=%d\n", v_is_device, r_is_device);
    exit(1);
  }
}

void SpMtv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  // Check if vectors are on device
  cudaPointerAttributes attrs_v, attrs_r;
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);
  cudaError_t err_r = cudaPointerGetAttributes(&attrs_r, r);

  // If cudaPointerGetAttributes fails, assume host pointer
  int v_is_device = (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice);
  int r_is_device = (err_r == cudaSuccess && attrs_r.type == cudaMemoryTypeDevice);

  // Use device matrix only if BOTH vectors are on device and device matrix exists
  if (v_is_device && r_is_device) {
    if (!M->d_csc) {
      fprintf(stderr, "Error in SpMtv_matrix: Vectors are on device but device matrix (d_csc) is NULL. "
                      "Matrix was likely created without device allocation. "
                      "This can happen if the matrix was created with A=NULL or if device allocation failed.\n");
      exit(1);
    }
    // Use device matrix structure - pass device pointer to structure
    SpMtv(M->d_csc, v, r);
  }
  else if (!v_is_device && !r_is_device) {
    // Both vectors on host, use host matrix
    SpMtv(M->csc, v, r);
  }
  else {
    // Mixed: one vector on device, one on host - this is an error
    fprintf(stderr, "Error in SpMtv_matrix: Input and output vectors are on different memory spaces. "
                    "v_is_device=%d, r_is_device=%d\n", v_is_device, r_is_device);
    exit(1);
  }
}

QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  // Detect memory space of x
  cudaPointerAttributes attrs_x;
  cudaError_t err_x = cudaPointerGetAttributes(&attrs_x, x);
  int x_is_device = (err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice);
  if (x_is_device) {
    // Copy to host and compute norm
    QOCOFloat* x_host = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
    CUDA_CHECK(cudaMemcpy(x_host, x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));

    QOCOFloat norm = 0.0;
    for (QOCOInt i = 0; i < n; ++i) {
      QOCOFloat xi = qoco_abs(x_host[i]);
      norm = qoco_max(norm, xi);
    }

    qoco_free(x_host);

    return norm;
  }
  else {
    QOCOFloat norm = 0.0;
    for (QOCOInt i = 0; i < n; ++i) {
      QOCOFloat xi = qoco_abs(x[i]);
      norm = qoco_max(norm, xi);
    }
    return norm;
  }
}
