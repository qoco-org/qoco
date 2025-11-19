/**
 * @file builtin_types.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines the vector and matrices for builtin linear algebra.
 */

#ifndef BUILTIN_TYPES_H
#define BUILTIN_TYPES_H

#include "common_linalg.h"
#include "definitions.h"
#include "qoco_linalg.h"

struct QOCOVectori_ {
  QOCOInt* data;
  QOCOInt len;
};

struct QOCOVectorf_ {
  QOCOFloat* data;
  QOCOInt len;
};

struct QOCOMatrix_ {
  QOCOCscMatrix* csc;
};

#endif /* ifndef BUILTIN_TYPES_H */
