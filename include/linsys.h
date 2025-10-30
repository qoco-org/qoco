/**
 * @file linsys.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Prototype struct for linsys backends to implement.
 */

#ifndef LINSYS_H
#define LINSYS_H

#include "definitions.h"

typedef struct {
  void (*linsys_setup)();
  void (*linsys_factor)();
  void (*linsys_solve)();
} LinSysBackend;

#endif // LINSYS_H
