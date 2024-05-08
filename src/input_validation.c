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
  // max_iters must be positive.
  if (settings->max_iters <= 0) {
    return 1;
    printf("Settings validation error: max_iters must be positive.");
  }

  // max_iter_bisection must be positive.
  if (settings->max_iter_bisection <= 0) {
    return 1;
    printf("Settings validation error: max_iter_bisection must be positive.");
  }

  // abstol must be positive.
  if (settings->abstol <= 0) {
    return 1;
    printf("Settings validation error: abstol must be positive.");
  }

  // reltol must be non-negative.
  if (settings->reltol < 0) {
    return 1;
    printf("Settings validation error: reltol must be positive.");
  }

  return 0;
}

QCOSInt qcos_validate_data(const QCOSCscMatrix* P, const QCOSFloat* c,
                           const QCOSCscMatrix* A, const QCOSFloat* b,
                           const QCOSCscMatrix* G, const QCOSFloat* h,
                           const QCOSInt l, const QCOSInt ncones,
                           const QCOSInt* q)
{

  // If there are second-order cones, then the cone dimensions must be provided.
  if (!q && ncones != 0) {
    printf("Data validation error: Provide second-order cone dimensions.");
    return 1;
  }

  // P must be a square matrix.
  if (P->m != P->n) {
    printf("Data validation error: P must be a square matrix.");
    return 1;
  }

  // Number of columns for A and G must be n.
  if ((G->n != A->n) || (P->n != G->n)) {
    printf("Data validation error: The number of columns for A and G must be "
           "equal to n.");
    return 1;
  }

  // c cannot be null.
  if (!c) {
    printf("Data validation error: linear cost term, c, must be provided.");
  }

  // l + sum(q) should be equal to m.
  QCOSInt sum = l;
  for (QCOSInt i = 0; i < ncones; ++i) {
    sum += q[i];
  }
  if (sum != G->m) {
    printf("Data validation error: l + sum(q) must be equal to m.");
    return 1;
  }

  // l must be non-negative.
  if (l < 0) {
    printf("Data validation error: l must be non-negative.");
    return 1;
  }

  // ncones must be non-negative.
  if (ncones < 0) {
    printf("Data validation error: ncones must be non-negative.");
    return 1;
  }

  if ((A && !b) || (b && !A)) {
    printf("Data validation error: If there are equality constraints, A and b "
           "must be provided.");
  }

  if ((G && !h) || (h && !G)) {
    printf("Data validation error: If there are conic constraints, G and h "
           "must be provided.");
  }

  return 0;
}
