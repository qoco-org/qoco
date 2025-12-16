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
  QOCOCscMatrix* d_csc;    // Device pointer to device CSC matrix structure
};

#endif /* ifndef CUDA_TYPES_H */
