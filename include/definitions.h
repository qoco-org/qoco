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
 * Defines QCOSInt, QCOSFloat, qcos_malloc, qcos_calloc, and qcos_free.
 */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// Define QCOSInt and QCOSFloat.
#ifdef QCOS_LONG_INT
typedef long long QCOSInt;
#else
typedef int QCOSInt;
#endif

#ifdef QCOS_SINGLE_PRECISION
typedef float QCOSFloat;
#else
typedef double QCOSFloat;
#endif

#define qcos_max(a, b) (a > b) ? a : b
#include "math.h"
#define qcos_sqrt(a) sqrt(a)

// Need for malloc, calloc, and free.
#include <stdlib.h>
#define qcos_malloc malloc
#define qcos_calloc calloc
#define qcos_free free
#endif /* ifndef DEFINITIONS_H */