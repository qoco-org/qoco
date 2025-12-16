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

  if (A) {
    QOCOInt m = A->m;
    QOCOInt n = A->n;
    QOCOInt nnz = A->nnz;

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
  free_qoco_csc_matrix(A->csc);
  qoco_free(A);
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

QOCOFloat* get_data_vectorf(const QOCOVectorf* x)
{
  // During equilibration/setup (CPU phase), return host pointer
  // During solve (GPU phase), return device pointer to avoid CPU-GPU copies
  // if (x->d_data) {
  //   return x->d_data;
  // }
  return x->data;
}

QOCOInt get_length_vectorf(const QOCOVectorf* x) { return x->len; }

void sync_vector_to_host(QOCOVectorf* v)
{
  if (v && v->data && v->d_data) {
    CUDA_CHECK(cudaMemcpy(v->data, v->d_data, v->len * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToHost));
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

  // If either pointer check succeeds and one is on device, use CUDA kernel.
  if ((err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice) ||
      (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice)) {
    copy_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y, n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    // If both pointers are on host, use CPU.
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

  // If either pointer check succeeds and one is on device, use CUDA kernel.
  if ((err_x == cudaSuccess && attrs_x.type == cudaMemoryTypeDevice) ||
      (err_y == cudaSuccess && attrs_y.type == cudaMemoryTypeDevice)) {
    copy_and_negate_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y,
                                                            n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    // If both pointers are on host, use CPU.
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

  // Check if pointers are on device - handle errors gracefully
  cudaPointerAttributes attrs_u, attrs_v;
  cudaError_t err_u = cudaPointerGetAttributes(&attrs_u, u);
  cudaError_t err_v = cudaPointerGetAttributes(&attrs_v, v);

  // If either pointer check fails or one is on device, use CUDA
  if ((err_u == cudaSuccess && attrs_u.type == cudaMemoryTypeDevice) ||
      (err_v == cudaSuccess && attrs_v.type == cudaMemoryTypeDevice)) {
    // Ensure CUDA libraries are loaded
    if (!load_cuda_libraries()) {
      fprintf(stderr, "Failed to load CUDA libraries in qoco_dot\n");
      // Fallback to CPU implementation
      QOCOFloat x = 0.0;
      for (QOCOInt i = 0; i < n; ++i) {
        x += u[i] * v[i];
      }
      return x;
    }
    
    CudaLibFuncs* funcs = get_cuda_funcs();
    cublasHandle_t handle;
    funcs->cublasCreate(&handle);
    QOCOFloat result;
    funcs->cublasDdot(handle, n, (const double*)u, 1, (const double*)v, 1, (double*)&result);
    funcs->cublasDestroy(handle);
    return result;
  }
  else {
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

void USpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

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

void SpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  for (QOCOInt i = 0; i < M->m; ++i) {
    r[i] = 0.0;
  }

  for (QOCOInt j = 0; j < M->n; j++) {
    for (QOCOInt i = M->p[j]; i < M->p[j + 1]; i++) {
      r[M->i[i]] += M->x[i] * v[j];
    }
  }
}

void SpMtv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  for (QOCOInt i = 0; i < M->n; ++i) {
    r[i] = 0.0;
  }

  for (QOCOInt i = 0; i < M->n; i++) {
    for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
      r[i] += M->x[j] * v[M->i[j]];
    }
  }
}

void USpMv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  USpMv(M->csc, v, r);
}

void SpMv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  SpMv(M->csc, v, r);
}

void SpMtv_matrix(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  SpMtv(M->csc, v, r);
}

QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  QOCOFloat norm = 0.0;
  QOCOFloat xi;
  for (QOCOInt i = 0; i < n; ++i) {
    xi = qoco_abs(x[i]);
    norm = qoco_max(norm, xi);
  }
  return norm;
}
