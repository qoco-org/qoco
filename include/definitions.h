/**
 * @file definitions.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines various macros used in qcos.
 */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// Define QCOSInt and QCOSFloat.
#include <limits.h>
typedef int QCOSInt;
#define QCOSInt_MAX INT_MAX

typedef double QCOSFloat;
#define QCOSFloat_MAX __DBL_MAX__

#define qcos_max(a, b) (((a) > (b)) ? (a) : (b))
#define qcos_min(a, b) (((a) < (b)) ? (a) : (b))
#define qcos_abs(a) (((a) > 0) ? (a) : (-a))
#define safe_div(a, b) (qcos_abs(a) > 1e-15) ? (a / b) : QCOSFloat_MAX
#include "math.h"
#define qcos_sqrt(a) sqrt(a)

#ifdef QCOS_DEBUG
#include <assert.h>
#include <stdio.h>
#define qcos_assert(a)                                                         \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Assertion Failed: %s\n", #a);                                    \
      __asm__ volatile("int $0x03");                                           \
    }                                                                          \
  } while (0)
#else
#define qcos_assert(a)                                                         \
  do {                                                                         \
  } while (0)
#endif

// Need for malloc, calloc, and free.
#include <stdlib.h>
#define qcos_malloc malloc
#define qcos_calloc calloc
#define qcos_free free
#endif /* ifndef DEFINITIONS_H */