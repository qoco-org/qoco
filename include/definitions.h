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
#ifdef QCOS_LONG_INT
typedef long long QCOSInt;
#define QCOSInt_MAX LLONG_MAX
#else
typedef int QCOSInt;
#define QCOSInt_MAX INT_MAX
#endif

#ifdef QCOS_SINGLE_PRECISION
typedef float QCOSFloat;
#define QCOSFloat_MAX __FLT_MAX__
#else
typedef double QCOSFloat;
#define QCOSFloat_MAX __DBL_MAX__

#endif

#define qcos_max(a, b) (a > b) ? a : b
#define qcos_min(a, b) (a < b) ? a : b
#define qcos_abs(a) (a > 0) ? a : -a
#define safe_div(a, b) (qcos_abs(a) > 1e-15) ? (a / b) : QCOSFloat_MAX
#include "math.h"
#define qcos_sqrt(a) sqrt(a)

// Need for malloc, calloc, and free.
#include <stdlib.h>
#define qcos_malloc malloc
#define qcos_calloc calloc
#define qcos_free free
#endif /* ifndef DEFINITIONS_H */