/**
 * @file definitions.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines various macros used in qoco.
 */

#ifndef QOCO_DEFINITIONS_H
#define QOCO_DEFINITIONS_H

#include "qoco_config.h"

// Define QOCOInt and QOCOFloat.
#include <float.h>
#include <limits.h>
typedef int QOCOInt;
#define QOCOInt_MAX INT_MAX

#ifdef MATLAB
#define printf mexPrintf
#endif

#include <math.h>
#ifdef QOCO_SINGLE_PRECISION
typedef float QOCOFloat;
#define QOCOFloat_MAX FLT_MAX
#define qoco_sqrt(a) sqrtf(a)
#define QOCOFloat_PRINT_FORMAT ".9g"
#define QOCOFloat_PRINT_ARG(a) ((double)(a))
#elif defined(QOCO_LONG_DOUBLE_PRECISION)
#if LDBL_MANT_DIG <= DBL_MANT_DIG
#error "QOCO_LONG_DOUBLE_PRECISION requires long double to be wider than double"
#endif
typedef long double QOCOFloat;
#define QOCOFloat_MAX LDBL_MAX
#define qoco_sqrt(a) sqrtl(a)
#define QOCOFloat_PRINT_FORMAT ".21Lg"
#define QOCOFloat_PRINT_ARG(a) ((long double)(a))
#else
typedef double QOCOFloat;
#define QOCOFloat_MAX DBL_MAX
#define qoco_sqrt(a) sqrt(a)
#define QOCOFloat_PRINT_FORMAT ".17g"
#define QOCOFloat_PRINT_ARG(a) ((double)(a))
#endif

#define qoco_max(a, b) (((a) > (b)) ? (a) : (b))
#define qoco_min(a, b) (((a) < (b)) ? (a) : (b))
#define qoco_abs(a) (((a) > 0) ? (a) : (-(a)))
#define safe_div(a, b)                                                        \
  ((qoco_abs(b) > (QOCOFloat)1e-15) ? ((a) / (b)) : QOCOFloat_MAX)

#if defined(QOCO_DEBUG) && defined(IS_LINUX)
#include <assert.h>
#include <stdio.h>
#define qoco_assert(a)                                                         \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Assertion Failed: %s\n", #a);                                    \
      __asm__ volatile("int $0x03");                                           \
    }                                                                          \
  } while (0)
#else
#define qoco_assert(a)                                                         \
  do {                                                                         \
  } while (0)
#endif

// Need for malloc, calloc, and free.
#include <stdlib.h>
#define qoco_malloc malloc
#define qoco_calloc calloc
#define qoco_free free
#endif /* ifndef QOCO_DEFINITIONS_H */