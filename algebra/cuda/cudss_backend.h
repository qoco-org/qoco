/**
 * @file cudss_backend.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines the cuDSS backend.
 */

#ifndef CUDSS_BACKEND_H
#define CUDSS_BACKEND_H

#include "cuda_types.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "common_linalg.h"
#include "kkt.h"
#ifdef __cplusplus
}
#endif
#include "structs.h"

#ifdef HAVE_CUDSS
#include <cudss.h>
#else
// Stub definitions for cuDSS types if not available
typedef void* cudssHandle_t;
typedef void* cudssConfig_t;
typedef void* cudssData_t;
typedef void* cudssMatrix_t;
typedef enum { CUDSS_STATUS_SUCCESS } cudssStatus_t;
typedef enum { CUDSS_PHASE_ANALYSIS, CUDSS_PHASE_FACTORIZATION, CUDSS_PHASE_SOLVE } cudssPhase_t;
#define cudssCreate(x) (CUDSS_STATUS_SUCCESS)
#define cudssConfigCreate(x) (CUDSS_STATUS_SUCCESS)
#define cudssDataCreate(h, x) (CUDSS_STATUS_SUCCESS)
#define cudssMatrixCreateCsr(h, m, nrows, ncols, nnz, rp, ci, v) (CUDSS_STATUS_SUCCESS)
#define cudssExecute(h, p, c, d, m, x, b) (CUDSS_STATUS_SUCCESS)
#define cudssMatrixDestroy(m) ((void)0)
#define cudssDataDestroy(d) ((void)0)
#define cudssConfigDestroy(c) ((void)0)
#define cudssDestroy(h) ((void)0)
#endif

extern LinSysBackend backend;

#endif /* #ifndef CUDSS_BACKEND_H */

