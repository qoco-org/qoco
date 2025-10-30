/**
 * @file qdldl_backend.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "qdldl_backend.h"
#include <stdio.h>

static void qdldl_setup() { printf("HI"); }

static void qdldl_factor() {}

static void qdldl_solve() {}

// Export the backend struct
LinSysBackend qdldl_backend = {.linsys_setup = qdldl_setup,
                               .linsys_factor = qdldl_factor,
                               .linsys_solve = qdldl_solve};
