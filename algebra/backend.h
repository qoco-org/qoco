/**
 * @file backend.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes the appropriate backend header based on QOCO_ALGEBRA_BACKEND.
 */

#ifndef QOCO_BACKEND_H
#define QOCO_BACKEND_H

#if defined(QOCO_ALGEBRA_BACKEND_BUILTIN) || !defined(QOCO_ALGEBRA_BACKEND_CUDA)
#include "builtin/qdldl_backend.h"
#elif defined(QOCO_ALGEBRA_BACKEND_CUDA)
#include "cuda/cudss_backend.h"
#else
#error "Unknown algebra backend. Define QOCO_ALGEBRA_BACKEND_BUILTIN or QOCO_ALGEBRA_BACKEND_CUDA"
#endif

#endif /* #ifndef QOCO_BACKEND_H */

