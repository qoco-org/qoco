/**
 * @file input_validation.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "input_validation.h"

QCOSInt qcos_validate_settings(QCOSSettings* settings)
{
  if (settings->max_iters < 0)
    return 1;
  return 0;
}

QCOSInt qcos_validate_data(const QCOSCscMatrix* P, const QCOSFloat* c,
                           const QCOSCscMatrix* A, const QCOSFloat* b,
                           const QCOSCscMatrix* G, const QCOSFloat* h,
                           const QCOSInt l, const QCOSInt ncones,
                           const QCOSInt* q)
{
  if (!P)
    return 1;
  if (!c)
    return 1;
  if (!A)
    return 1;
  if (!b)
    return 1;
  if (!G)
    return 1;
  if (!h)
    return 1;
  if (l < 0)
    return 1;
  if (ncones < 0)
    return 1;
  if (!q)
    return 1;

  return 0;
}
