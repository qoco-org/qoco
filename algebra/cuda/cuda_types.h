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
 * On GPU, matrices are stored in CSR format (as required by cuDSS).
 */

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

extern "C" {
#include "common_linalg.h"
#include "definitions.h"
#include "qoco_linalg.h"
}
#include <cuda_runtime.h>

struct QOCOVectori_ {
  QOCOInt* data;   // Host pointer
  QOCOInt* d_data; // Device pointer
  QOCOInt len;
};

struct QOCOVectorf_ {
  QOCOFloat* data;   // Host pointer
  QOCOFloat* d_data; // Device pointer
  QOCOInt len;
};

struct QOCOMatrix_ {
  QOCOCscMatrix* csc;      // Host CSC matrix
  QOCOCscMatrix* d_csc;    // Device CSC matrix (NULL if not allocated)
};

#endif /* ifndef CUDA_TYPES_H */
