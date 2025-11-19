/**
 * @file cuda_types.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines the vector and matrices for CUDA linear algebra.
 * On GPU, matrices are stored in CSR format (as required by cuSPARSE/cuDSS).
 */

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif
#include "common_linalg.h"
#include "definitions.h"
#include "qoco_linalg.h"
#ifdef __cplusplus
}
#endif
#include <cuda_runtime.h>

// CSR matrix format (for GPU storage)
typedef struct {
  /** Number of rows. */
  QOCOInt m;

  /** Number of columns. */
  QOCOInt n;

  /** Number of nonzero elements. */
  QOCOInt nnz;

  /** Row pointers (length: m+1). */
  QOCOInt* row_ptr;

  /** Column indices (length: nnz). */
  QOCOInt* col_ind;

  /** Data (length: nnz). */
  QOCOFloat* val;
} QOCOCsrMatrix;

struct QOCOVectori_ {
  QOCOInt* data;  // Host pointer (kept for compatibility, but not used during solve)
  QOCOInt* d_data;  // Device pointer (primary storage)
  QOCOInt len;
};

struct QOCOVectorf_ {
  QOCOFloat* data;  // Host pointer (kept for compatibility, but not used during solve)
  QOCOFloat* d_data;  // Device pointer (primary storage)
  QOCOInt len;
};

struct QOCOMatrix_ {
  QOCOCscMatrix* csc;  // Host CSC (used for setup/CPU operations)
  QOCOCsrMatrix* d_csr;  // Device CSR (primary GPU storage)
};

#endif /* ifndef CUDA_TYPES_H */
