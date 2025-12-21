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

__global__ void ew_product_kernel(const QOCOFloat* x, const QOCOFloat* y,
                                  QOCOFloat* z, QOCOInt n)
{
  QOCOInt i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = x[i] * y[i];
  }
}

__global__ void USpMv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v,
                             QOCOFloat* r)
{
  QOCOInt i = blockIdx.x;
  if (i >= M->n)
    return;

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

__global__ void SpMtv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v,
                             QOCOFloat* r)
{
  QOCOInt col = blockIdx.x;
  if (col >= M->n)
    return;

  QOCOFloat sum = 0.0;

  // Process all nonzeros in column col
  for (QOCOInt idx = M->p[col]; idx < M->p[col + 1]; idx++) {
    QOCOInt row = M->i[idx];
    sum += M->x[idx] * v[row];
  }

  r[col] = sum;
}

__global__ void SpMv_kernel(const QOCOCscMatrix* M, const QOCOFloat* v,
                            QOCOFloat* r)
{
  QOCOInt col = blockIdx.x;
  if (col >= M->n)
    return;

  // Process all nonzeros in column col
  for (QOCOInt idx = M->p[col]; idx < M->p[col + 1]; idx++) {
    QOCOInt row = M->i[idx];
    QOCOFloat val = M->x[idx] * v[col];

    // Add to r[row] using atomic (multiple columns can write to same row)
    atomicAdd(&r[row], val);
  }
}

// Construct A on CPU and create device copy.
QOCOMatrix* new_qoco_matrix(const QOCOCscMatrix* A)
{
  CUDA_CHECK(cudaGetLastError());
  QOCOMatrix* M = (QOCOMatrix*)qoco_malloc(sizeof(QOCOMatrix));

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

    // Allocate device CSC matrix (host copy for freeing)
    M->d_csc_host = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));

    QOCOCscMatrix* d = M->d_csc_host;
    d->m = m;
    d->n = n;
    d->nnz = nnz;

    CUDA_CHECK(cudaMalloc(&d->x, nnz * sizeof(QOCOFloat)));
    CUDA_CHECK(cudaMalloc(&d->i, nnz * sizeof(QOCOInt)));
    CUDA_CHECK(cudaMalloc(&d->p, (n + 1) * sizeof(QOCOInt)));

    CUDA_CHECK(
        cudaMemcpy(d->x, x, nnz * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d->i, i, nnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d->p, p, (n + 1) * sizeof(QOCOInt), cudaMemcpyHostToDevice));

    // Allocate device struct itself and copy host struct into device
    CUDA_CHECK(cudaMalloc(&M->d_csc, sizeof(QOCOCscMatrix)));
    CUDA_CHECK(
        cudaMemcpy(M->d_csc, d, sizeof(QOCOCscMatrix), cudaMemcpyHostToDevice));
  }
  else {
    M->csc = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));
    M->d_csc_host = (QOCOCscMatrix*)qoco_malloc(sizeof(QOCOCscMatrix));
    CUDA_CHECK(cudaMalloc(&M->d_csc, sizeof(QOCOCscMatrix)));

    M->csc->m = 0;
    M->csc->n = 0;
    M->csc->nnz = 0;
    M->csc->x = NULL;
    M->csc->i = NULL;
    M->csc->p = NULL;

    M->d_csc_host->m = 0;
    M->d_csc_host->n = 0;
    M->d_csc_host->nnz = 0;
    M->d_csc_host->x = NULL;
    M->d_csc_host->i = NULL;
    M->d_csc_host->p = NULL;
  }
  CUDA_CHECK(cudaGetLastError());

  return M;
}

// Construct x on CPU copy vector to GPU.
QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n)
{
  CUDA_CHECK(cudaGetLastError());
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

// Construct x on CPU copy vector to GPU.
QOCOVectori* new_qoco_vectori(const QOCOInt* x, QOCOInt n)
{
  QOCOVectori* v = (QOCOVectori*)qoco_malloc(sizeof(QOCOVectori));
  QOCOInt* vdata = (QOCOInt*)qoco_malloc(sizeof(QOCOInt) * n);

  if (x) {
    copy_arrayi(x, vdata, n);
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      vdata[i] = 0;
    }
  }

  QOCOInt* d_vdata;
  CUDA_CHECK(cudaMalloc(&d_vdata, sizeof(QOCOInt) * n));
  CUDA_CHECK(
      cudaMemcpy(d_vdata, vdata, sizeof(QOCOInt) * n, cudaMemcpyHostToDevice));

  v->len = n;
  v->data = vdata;
  v->d_data = d_vdata;

  return v;
}

void free_qoco_matrix(QOCOMatrix* A)
{
  CUDA_CHECK(cudaGetLastError());
  if (!A)
    return;

  // Free host CSC
  free_qoco_csc_matrix(A->csc);

  // Free device CSC
  if (A->d_csc_host) {
    CUDA_CHECK(cudaFree(A->d_csc_host->x));
    CUDA_CHECK(cudaFree(A->d_csc_host->i));
    CUDA_CHECK(cudaFree(A->d_csc_host->p));
    qoco_free(A->d_csc_host);
  }

  if (A->d_csc) {
    CUDA_CHECK(cudaFree(A->d_csc));
  }

  qoco_free(A);
  CUDA_CHECK(cudaGetLastError());
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

void free_qoco_vectori(QOCOVectori* x)
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

QOCOInt get_length_vectorf(const QOCOVectorf* x) { return x->len; }

void sync_vector_to_host(QOCOVectorf* v)
{
  if (v && v->data && v->d_data) {
    CUDA_CHECK(cudaMemcpy(v->data, v->d_data, v->len * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToHost));
  }
}

void sync_vector_to_device(QOCOVectorf* v)
{
  if (v && v->data && v->d_data) {
    CUDA_CHECK(cudaMemcpy(v->d_data, v->data, v->len * sizeof(QOCOFloat),
                          cudaMemcpyHostToDevice));
  }
}

void sync_matrix_to_device(QOCOMatrix* M)
{
  if (M->csc && M->d_csc) {
    CUDA_CHECK(cudaMemcpy(M->d_csc_host->x, M->csc->x,
                          M->csc->nnz * sizeof(QOCOFloat),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M->d_csc_host->i, M->csc->i,
                          M->csc->nnz * sizeof(QOCOInt),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M->d_csc_host->p, M->csc->p,
                          (M->csc->n + 1) * sizeof(QOCOInt),
                          cudaMemcpyHostToDevice));
  }
}

static int in_cpu_mode = 0;

void set_cpu_mode(int active) { in_cpu_mode = active; }

QOCOFloat* get_data_vectorf(const QOCOVectorf* x)
{
  CUDA_CHECK(cudaGetLastError());
  if (in_cpu_mode) {
    return x->data;
  }
  else {
    return x->d_data;
  }
}

QOCOFloat get_element_vectorf(const QOCOVectorf* x, QOCOInt idx)
{
  if (in_cpu_mode) {
    return x->data[idx];
  }
  else {
    return x->d_data[idx];
  }
}

QOCOFloat* get_pointer_vectorf(const QOCOVectorf* x, QOCOInt idx)
{
  if (in_cpu_mode) {
    return &x->data[idx];
  }
  else {
    return &x->d_data[idx];
  }
}

QOCOInt* get_data_vectori(const QOCOVectori* x)
{
  if (in_cpu_mode) {
    return x->data;
  }
  else {
    return x->d_data;
  }
}

QOCOInt get_element_vectori(const QOCOVectori* x, QOCOInt idx)
{
  if (in_cpu_mode) {
    return x->data[idx];
  }
  else {
    return x->d_data[idx];
  }
}

QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M)
{
  if (in_cpu_mode) {
    return M->csc;
  }
  else {
    return M->d_csc_host;
  }
}

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
  CUDA_CHECK(cudaGetLastError());
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

  if (n <= 0) {
    return;
  }

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
    CUDA_CHECK(cudaGetLastError());
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
  CUDA_CHECK(cudaGetLastError());

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
    CUDA_CHECK(cudaGetLastError());
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

void ew_product(QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z, QOCOInt n)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  if (n > 0) {
    ew_product_kernel<<<blocks, threads>>>(x, y, z, n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}

// TODO: Don't create and destroy cublas handle for each dot product. One way
// around this is to create a custom kernel for the dot product.
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

    CudaLibFuncs* funcs = get_cuda_funcs();
    cublasHandle_t handle;
    funcs->cublasCreate(&handle);
    QOCOFloat result;
    funcs->cublasDdot(handle, n, (const double*)u, 1, (const double*)v, 1,
                      (double*)&result);
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
      CUDA_CHECK(cudaGetLastError());
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
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    for (QOCOInt i = 0; i < n; ++i) {
      z[i] = a * x[i] + y[i];
    }
  }
}

void USpMv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  if (M->d_csc_host->n > 0) {
    CUDA_CHECK(cudaMemset(r, 0, M->d_csc_host->n * sizeof(QOCOFloat)));
    USpMv_kernel<<<M->d_csc_host->n, 1>>>(M->d_csc, v, r);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void SpMv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  if (M->d_csc_host->m > 0) {
    CUDA_CHECK(cudaMemset(r, 0, M->d_csc_host->m * sizeof(QOCOFloat)));
    SpMv_kernel<<<M->d_csc_host->n, 1>>>(M->d_csc, v, r);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void SpMtv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  if (M->d_csc_host->n > 0) {
    CUDA_CHECK(cudaMemset(r, 0, M->d_csc_host->n * sizeof(QOCOFloat)));
    SpMtv_kernel<<<M->d_csc_host->n, 1>>>(M->d_csc, v, r);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  if (n == 0)
    return 0.0;

  // Check if pointer is on device - handle errors gracefully
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, x);

  if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
    // Load CUDA libraries dynamically. TODO: This should be done in qoco_setup.
    if (!load_cuda_libraries()) {
      fprintf(stderr, "Failed to load CUDA libraries\n");
      exit(1);
    }
    CudaLibFuncs* funcs = get_cuda_funcs();
    cublasHandle_t handle;
    funcs->cublasCreate(&handle);

    int idx_max;
    QOCOFloat max_val;

    funcs->cublasIdamax(handle, n, (const double*)x, 1, &idx_max);
    // Note: cuBLAS uses 1-based indexing, so subtract 1
    idx_max -= 1;
    CUDA_CHECK(cudaMemcpy(&max_val, &x[idx_max], sizeof(QOCOFloat),
                          cudaMemcpyDeviceToHost));

    funcs->cublasDestroy(handle);
    return qoco_abs(max_val);
  }
  else {
    QOCOFloat norm = 0.0;
    QOCOFloat xi;
    for (QOCOInt i = 0; i < n; ++i) {
      xi = qoco_abs(x[i]);
      norm = qoco_max(norm, xi);
    }
    return norm;
  }
}

QOCOFloat min_abs_val(const QOCOFloat* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  if (n == 0)
    return QOCOFloat_MAX;

  // Check if pointer is on device - handle errors gracefully
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, x);

  // If pointer check fails, it might be a host pointer or CUDA not initialized
  // If it succeeds and is on device, use cuBLAS
  if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
    // Load CUDA libraries dynamically. TODO: This should be done in qoco_setup.
    if (!load_cuda_libraries()) {
      fprintf(stderr, "Failed to load CUDA libraries\n");
      exit(1);
    }
    CudaLibFuncs* funcs = get_cuda_funcs();
    cublasHandle_t handle;
    funcs->cublasCreate(&handle);

    int idx_min;
    QOCOFloat min_val;

    funcs->cublasIdamin(handle, n, (const double*)x, 1, &idx_min);
    // Note: cuBLAS uses 1-based indexing, so subtract 1
    idx_min -= 1;
    CUDA_CHECK(cudaMemcpy(&min_val, &x[idx_min], sizeof(QOCOFloat),
                          cudaMemcpyDeviceToHost));

    funcs->cublasDestroy(handle);
    return qoco_abs(min_val);
  }
  else {
    QOCOFloat min_val = QOCOFloat_MAX;
    QOCOFloat xi;
    for (QOCOInt i = 0; i < n; ++i) {
      xi = qoco_abs(x[i]);
      min_val = qoco_min(min_val, xi);
    }
    return min_val;
  }
}

QOCOInt check_nan(const QOCOVectorf* x)
{
  for (QOCOInt i = 0; i < x->len; ++i) {
    if (isnan(x->data[i])) {
      return 1;
    }
  }
  return 0;
}
