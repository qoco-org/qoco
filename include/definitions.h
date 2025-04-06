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
#include <limits.h>
typedef int QOCOInt;
#define QOCOInt_MAX INT_MAX

#ifdef MATLAB
#define printf mexPrintf
#endif

typedef double QOCOFloat;
#ifdef IS_WINDOWS
#define QOCOFloat_MAX 1e308
#else
#define QOCOFloat_MAX __DBL_MAX__
#endif

#define qoco_max(a, b) (((a) > (b)) ? (a) : (b))
#define qoco_min(a, b) (((a) < (b)) ? (a) : (b))
#define qoco_abs(a) (((a) > 0) ? (a) : (-a))
#define safe_div(a, b) (qoco_abs(b) > 1e-15) ? (a / b) : QOCOFloat_MAX
#include <math.h>
#define qoco_sqrt(a) sqrt(a)

#if defined(QOCO_DEBUG) && !defined(IS_WINDOWS)
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