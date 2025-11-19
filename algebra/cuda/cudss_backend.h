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
#include <cudss.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "common_linalg.h"
#include "kkt.h"
#ifdef __cplusplus
}
#endif
#include "structs.h"

extern LinSysBackend backend;

#endif /* #ifndef CUDSS_BACKEND_H */
