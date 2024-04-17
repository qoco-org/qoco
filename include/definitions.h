#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// Needed for int64_t and int32_t datatypes
#include "stdint.h"

/* Define QCOS ints and floats */
#ifdef QCOS_LONG_INT
typedef int64_t QCOSInt;
#else
typedef int32_t QCOSInt;
#endif

#ifdef QCOS_SINGLE_PRECISION
typedef float QCOSFloat;
#else
typedef double QCOSFloat;
#endif

#include <stdlib.h>
#define qcos_malloc malloc
#define qcos_calloc calloc
#define qcos_free free
#endif /* ifndef DEFINITIONS_H */