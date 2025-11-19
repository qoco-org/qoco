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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

extern "C" {
#include "common_linalg.h"
}

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while(0)

// CUDA kernels
__global__ void copy_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y, QOCOInt n) {
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

__global__ void copy_and_negate_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y, QOCOInt n) {
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = -x[idx];
  }
}

__global__ void copy_arrayi_kernel(const QOCOInt* x, QOCOInt* y, QOCOInt n) {
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

__global__ void scale_arrayf_kernel(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n) {
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = s * x[idx];
  }
}

__global__ void axpy_kernel(const QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z, QOCOFloat a, QOCOInt n) {
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = a * x[idx] + y[idx];
  }
}

__global__ void inf_norm_kernel(const QOCOFloat* x, QOCOFloat* result, QOCOInt n) {
  extern __shared__ QOCOFloat sdata[];
  QOCOInt tid = threadIdx.x;
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  QOCOFloat val = (idx < n) ? fabs(x[idx]) : 0.0f;
  sdata[tid] = val;
  __syncthreads();
  
  for (QOCOInt s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

static inline QOCOInt get_block_size(QOCOInt n) {
  return (n + 255) / 256 * 256 < 256 ? 256 : ((n + 255) / 256) * 256;
}

// Convert CSC to CSR format (host conversion, then copy to device)
// Used for converting matrices to GPU format after setup/equilibration
static void csc_to_csr(const QOCOCscMatrix* csc, QOCOInt** csr_row_ptr, 
                       QOCOInt** csr_col_ind, QOCOFloat** csr_val)
{
  QOCOInt m = csc->m;
  QOCOInt n = csc->n;
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
  
  // Allocate device memory and copy
  CUDA_CHECK(cudaMalloc(csr_row_ptr, (m + 1) * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMalloc(csr_col_ind, nnz * sizeof(QOCOInt)));
  CUDA_CHECK(cudaMalloc(csr_val, nnz * sizeof(QOCOFloat)));
  
  CUDA_CHECK(cudaMemcpy(*csr_row_ptr, h_csr_row_ptr, (m + 1) * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_col_ind, h_csr_col_ind, nnz * sizeof(QOCOInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*csr_val, h_csr_val, nnz * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  
  // Free host memory
  qoco_free(h_csr_row_ptr);
  qoco_free(h_csr_col_ind);
  qoco_free(h_csr_val);
}

extern "C" {

// Host implementation that manages GPU memory
QOCOMatrix* new_qoco_matrix(const QOCOCscMatrix* A)
{
  QOCOMatrix* M = (QOCOMatrix*)qoco_malloc(sizeof(QOCOMatrix));
  
  if (A) {
    QOCOInt m = A->m;
    QOCOInt n = A->n;
    QOCOInt nnz = A->nnz;

    // Allocate host memory
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

    // For problem matrices (P, A, G), don't convert to CSR yet
    // Equilibration happens on CPU using CSC, then KKT matrix is formed
    // Only KKT matrix needs to be converted to CSR for cuDSS
    // So for now, leave d_csr as NULL - it will be set up when KKT matrix is created
    M->csc = Mcsc;
    M->d_csr = NULL;
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
    M->d_csr = NULL;
  }

  return M;
}

QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n)
{
  QOCOVectorf* v = (QOCOVectorf*)qoco_malloc(sizeof(QOCOVectorf));
  QOCOFloat* vdata = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * n);
  
  if (x) {
    copy_arrayf(x, vdata, n);
  } else {
    for (QOCOInt i = 0; i < n; ++i) {
      vdata[i] = 0.0;
    }
  }

  QOCOFloat* d_vdata;
  CUDA_CHECK(cudaMalloc(&d_vdata, sizeof(QOCOFloat) * n));
  CUDA_CHECK(cudaMemcpy(d_vdata, vdata, sizeof(QOCOFloat) * n, cudaMemcpyHostToDevice));

  v->len = n;
  v->data = vdata;
  v->d_data = d_vdata;

  return v;
}

void free_qoco_matrix(QOCOMatrix* A)
{
  if (A) {
    if (A->csc) {
      free_qoco_csc_matrix(A->csc);
    }
    if (A->d_csr) {
      if (A->d_csr->row_ptr) cudaFree(A->d_csr->row_ptr);
      if (A->d_csr->col_ind) cudaFree(A->d_csr->col_ind);
      if (A->d_csr->val) cudaFree(A->d_csr->val);
      qoco_free(A->d_csr);
    }
    qoco_free(A);
  }
}

void free_qoco_vectorf(QOCOVectorf* x)
{
  if (x) {
    if (x->data) qoco_free(x->data);
    if (x->d_data) cudaFree(x->d_data);
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
  // Return device pointer for GPU operations
  return x->d_data;
}

QOCOInt get_length_vectorf(const QOCOVectorf* x)
{
  return x->len;
}

QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M)
{
  return M->csc;
}

// These functions are only called during equilibration on CPU, so call C functions directly
void col_inf_norm_USymm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during setup on CPU - use existing C function
  col_inf_norm_USymm(get_csc_matrix(M), norm);
}

void col_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during setup on CPU - use existing C function
  col_inf_norm(get_csc_matrix(M), norm);
}

void row_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm)
{
  // Called during setup on CPU - use existing C function
  row_inf_norm(get_csc_matrix(M), norm);
}

void row_col_scale_matrix(QOCOMatrix* M, const QOCOFloat* E, const QOCOFloat* D)
{
  // Called during setup on CPU - scale on host CSC
  row_col_scale(get_csc_matrix(M), (QOCOFloat*)E, (QOCOFloat*)D);
  
  // Clear device CSR if it exists - it will be recreated from updated CSC when KKT matrix is formed
  if (M->d_csr) {
    // Free old device CSR - it will be recreated from updated CSC later
    if (M->d_csr->row_ptr) cudaFree(M->d_csr->row_ptr);
    if (M->d_csr->col_ind) cudaFree(M->d_csr->col_ind);
    if (M->d_csr->val) cudaFree(M->d_csr->val);
    qoco_free(M->d_csr);
    M->d_csr = NULL;
  }
}

void set_element_vectorf(QOCOVectorf* x, QOCOInt idx, QOCOFloat data)
{
  // Update device directly (no host copy needed during solve)
  CUDA_CHECK(cudaMemcpy(&x->d_data[idx], &data, sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  x->data[idx] = data;  // Keep host in sync for compatibility
}

void reciprocal_vectorf(const QOCOVectorf* input, QOCOVectorf* output)
{
  // Operate directly on device
  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (input->len + blockSize - 1) / blockSize;
  
  // Use a kernel for reciprocal
  // For now, do on host then copy (setup phase only)
  for (QOCOInt i = 0; i < input->len; ++i) {
    output->data[i] = safe_div(1.0, input->data[i]);
  }
  CUDA_CHECK(cudaMemcpy(output->d_data, output->data, output->len * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
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

  if (n == 0) return;
  
  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;
  
  // Check if pointers are on device
  cudaPointerAttributes attrs_x, attrs_y;
  cudaPointerGetAttributes(&attrs_x, x);
  cudaPointerGetAttributes(&attrs_y, y);
  
  if (attrs_x.type == cudaMemoryTypeDevice || attrs_y.type == cudaMemoryTypeDevice) {
    // At least one pointer is on device - use kernel
    if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeDevice) {
      // Both device
      copy_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y, n);
    } else if (attrs_x.type == cudaMemoryTypeDevice) {
      // x on device, y on host
      CUDA_CHECK(cudaMemcpy(y, x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
    } else {
      // y on device, x on host
      CUDA_CHECK(cudaMemcpy((QOCOFloat*)y, x, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
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

  if (n == 0) return;
  
  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;
  
  cudaPointerAttributes attrs_x, attrs_y;
  cudaPointerGetAttributes(&attrs_x, x);
  cudaPointerGetAttributes(&attrs_y, y);
  
  if (attrs_x.type == cudaMemoryTypeDevice || attrs_y.type == cudaMemoryTypeDevice) {
    if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeDevice) {
      copy_and_negate_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y, n);
    } else if (attrs_x.type == cudaMemoryTypeDevice) {
        QOCOFloat* temp = (QOCOFloat*)qoco_malloc(n * sizeof(QOCOFloat));
      CUDA_CHECK(cudaMemcpy(temp, x, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
      for (QOCOInt i = 0; i < n; ++i) y[i] = -temp[i];
      qoco_free(temp);
    } else {
      QOCOFloat* d_y;
      CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(QOCOFloat)));
      CUDA_CHECK(cudaMemcpy(d_y, x, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
      copy_and_negate_arrayf_kernel<<<numBlocks, blockSize>>>(d_y, (QOCOFloat*)y, n);
      cudaFree(d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = -x[i];
    }
  }
}

void copy_arrayi(const QOCOInt* x, QOCOInt* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  if (n == 0) return;
  
  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;
  
  cudaPointerAttributes attrs_x, attrs_y;
  cudaPointerGetAttributes(&attrs_x, x);
  cudaPointerGetAttributes(&attrs_y, y);
  
  if (attrs_x.type == cudaMemoryTypeDevice || attrs_y.type == cudaMemoryTypeDevice) {
    if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeDevice) {
      copy_arrayi_kernel<<<numBlocks, blockSize>>>(x, (QOCOInt*)y, n);
    } else if (attrs_x.type == cudaMemoryTypeDevice) {
      CUDA_CHECK(cudaMemcpy(y, x, n * sizeof(QOCOInt), cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(cudaMemcpy((QOCOInt*)y, x, n * sizeof(QOCOInt), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    for (QOCOInt i = 0; i < n; ++i) {
      y[i] = x[i];
    }
  }
}

QOCOFloat qoco_dot(const QOCOFloat* u, const QOCOFloat* v, QOCOInt n)
{
  qoco_assert(u || n == 0);
  qoco_assert(v || n == 0);

  if (n == 0) return 0.0;

  cudaPointerAttributes attrs_u, attrs_v;
  cudaPointerGetAttributes(&attrs_u, u);
  cudaPointerGetAttributes(&attrs_v, v);

  if (attrs_u.type == cudaMemoryTypeDevice || attrs_v.type == cudaMemoryTypeDevice) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    QOCOFloat result;
    cublasDdot(handle, n, u, 1, v, 1, &result);
    cublasDestroy(handle);
    return result;
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
  qoco_assert(x || n == 0);

  if (n == 0) return -QOCOInt_MAX;

  cudaPointerAttributes attrs;
  cudaPointerGetAttributes(&attrs, x);

  if (attrs.type == cudaMemoryTypeDevice) {
    // Copy to host and compute
      QOCOInt* h_x = (QOCOInt*)qoco_malloc(n * sizeof(QOCOInt));
    CUDA_CHECK(cudaMemcpy(h_x, x, n * sizeof(QOCOInt), cudaMemcpyDeviceToHost));
    QOCOInt max = -QOCOInt_MAX;
    for (QOCOInt i = 0; i < n; ++i) {
      max = qoco_max(max, h_x[i]);
    }
    qoco_free(h_x);
    return max;
  } else {
    QOCOInt max = -QOCOInt_MAX;
    for (QOCOInt i = 0; i < n; ++i) {
      max = qoco_max(max, x[i]);
    }
    return max;
  }
}

void scale_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  if (n == 0) return;

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  cudaPointerAttributes attrs_x, attrs_y;
  cudaPointerGetAttributes(&attrs_x, x);
  cudaPointerGetAttributes(&attrs_y, y);

  if (attrs_x.type == cudaMemoryTypeDevice || attrs_y.type == cudaMemoryTypeDevice) {
    if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeDevice) {
      scale_arrayf_kernel<<<numBlocks, blockSize>>>(x, (QOCOFloat*)y, s, n);
    } else {
      cublasHandle_t handle;
      cublasCreate(&handle);
      if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeHost) {
        QOCOFloat* d_y;
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(QOCOFloat)));
        cublasDcopy(handle, n, x, 1, d_y, 1);
        cublasDscal(handle, n, &s, d_y, 1);
        CUDA_CHECK(cudaMemcpy(y, d_y, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
        cudaFree(d_y);
      } else {
        QOCOFloat* d_x;
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(QOCOFloat)));
        CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
        cublasDcopy(handle, n, d_x, 1, (QOCOFloat*)y, 1);
        cublasDscal(handle, n, &s, (QOCOFloat*)y, 1);
        cudaFree(d_x);
      }
      cublasDestroy(handle);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
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

  if (n == 0) return;

  const QOCOInt blockSize = 256;
  const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;

  cudaPointerAttributes attrs_x, attrs_y, attrs_z;
  cudaPointerGetAttributes(&attrs_x, x);
  cudaPointerGetAttributes(&attrs_y, y);
  cudaPointerGetAttributes(&attrs_z, z);

  if (attrs_x.type == cudaMemoryTypeDevice || attrs_y.type == cudaMemoryTypeDevice || attrs_z.type == cudaMemoryTypeDevice) {
    if (attrs_x.type == cudaMemoryTypeDevice && attrs_y.type == cudaMemoryTypeDevice && attrs_z.type == cudaMemoryTypeDevice) {
      axpy_kernel<<<numBlocks, blockSize>>>(x, y, (QOCOFloat*)z, a, n);
    } else {
      cublasHandle_t handle;
      cublasCreate(&handle);
      // For mixed cases, use cublas
      if (attrs_z.type == cudaMemoryTypeDevice) {
        cublasDcopy(handle, n, y, 1, (QOCOFloat*)z, 1);
        cublasDaxpy(handle, n, &a, x, 1, (QOCOFloat*)z, 1);
      } else {
        // Need to copy to host
        QOCOFloat* d_z;
        CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(QOCOFloat)));
        const QOCOFloat* d_x = (attrs_x.type == cudaMemoryTypeDevice) ? x : NULL;
        const QOCOFloat* d_y = (attrs_y.type == cudaMemoryTypeDevice) ? y : NULL;
        if (!d_x) {
          CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(QOCOFloat)));
          CUDA_CHECK(cudaMemcpy((void*)d_x, x, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
        }
        if (!d_y) {
          CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(QOCOFloat)));
          CUDA_CHECK(cudaMemcpy((void*)d_y, y, n * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
        }
        cublasDcopy(handle, n, d_y, 1, d_z, 1);
        cublasDaxpy(handle, n, &a, d_x, 1, d_z, 1);
        CUDA_CHECK(cudaMemcpy(z, d_z, n * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
        if (attrs_x.type == cudaMemoryTypeHost) cudaFree((void*)d_x);
        if (attrs_y.type == cudaMemoryTypeHost) cudaFree((void*)d_y);
        cudaFree(d_z);
      }
      cublasDestroy(handle);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    for (QOCOInt i = 0; i < n; ++i) {
      z[i] = a * x[i] + y[i];
    }
  }
}

// Sparse matrix-vector multiplication using cuSPARSE
void USpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // For now, use CPU implementation
  // TODO: Implement CUDA version using cuSPARSE
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

} // extern "C" - end of C linkage block

// Functions that don't need C linkage (called from C++ code or headers without extern "C")
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

  if (n == 0) return 0.0;

  cudaPointerAttributes attrs;
  cudaPointerGetAttributes(&attrs, x);

  if (attrs.type == cudaMemoryTypeDevice) {
    const QOCOInt blockSize = 256;
    const QOCOInt numBlocks = (n + blockSize - 1) / blockSize;
    QOCOFloat* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, numBlocks * sizeof(QOCOFloat)));
    
    inf_norm_kernel<<<numBlocks, blockSize, blockSize * sizeof(QOCOFloat)>>>(x, d_result, n);
    
    QOCOFloat* h_result = (QOCOFloat*)qoco_malloc(numBlocks * sizeof(QOCOFloat));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, numBlocks * sizeof(QOCOFloat), cudaMemcpyDeviceToHost));
    
    QOCOFloat result = 0.0;
    for (QOCOInt i = 0; i < numBlocks; ++i) {
      result = qoco_max(result, h_result[i]);
    }
    
    qoco_free(h_result);
    cudaFree(d_result);
    return result;
  } else {
    QOCOFloat norm = 0.0;
    QOCOFloat xi;
    for (QOCOInt i = 0; i < n; ++i) {
      xi = qoco_abs(x[i]);
      norm = qoco_max(norm, xi);
    }
    return norm;
  }
}
