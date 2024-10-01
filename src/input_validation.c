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

QCOSInt qcos_validate_settings(const QCOSSettings* settings)
{
  // max_iters must be positive.
  if (settings->max_iters <= 0) {
    printf("max_iters must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // ruiz_iters must be positive.
  if (settings->ruiz_iters < 0) {
    printf("ruiz_iters must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // bisection_iters must be positive.
  if (settings->bisection_iters <= 0) {
    printf("bisection_iters must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // abstol must be positive.
  if (settings->abstol <= 0) {
    printf("abstol must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // reltol must be non-negative.
  if (settings->reltol < 0) {
    printf("reltol must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // abstol_inaccurate must be positive.
  if (settings->abstol_inaccurate <= 0) {
    printf("abstol_inaccurate must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // reltol_inaccurate must be non-negative.
  if (settings->reltol_inaccurate < 0) {
    printf("reltol must be positive.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // static_reg must be less than 1.
  if (settings->static_reg > 1) {
    printf("static_reg must be less than 1.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  // dyn_reg must be less than 1.
  if (settings->dyn_reg > 1) {
    printf("dyn_reg must be less than 1.\n");
    return QCOS_SETTINGS_VALIDATION_ERROR;
  }

  return QCOS_NO_ERROR;
}

QCOSInt qcos_validate_data(const QCOSCscMatrix* P, const QCOSFloat* c,
                           const QCOSCscMatrix* A, const QCOSFloat* b,
                           const QCOSCscMatrix* G, const QCOSFloat* h,
                           const QCOSInt l, const QCOSInt nsoc,
                           const QCOSInt* q)
{

  // If there are second-order cones, then the cone dimensions must be provided.
  if (!q && nsoc != 0) {
    printf("Data validation error: Provide second-order cone dimensions.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // P must be a square matrix.
  if (P && P->m != P->n) {
    printf("Data validation error: P must be a square matrix.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // Number of columns for A and G must be equal.
  if (G && A && (G->n != A->n)) {
    printf("Data validation error: The number of columns for A and G must be "
           "equal to n.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // Number of columns for A must be equal to n.
  if (A && P && (P->n != A->n)) {
    printf("Data validation error: The number of columns for A must be "
           "equal to n.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // Number of columns for G must be equal to n.
  if (G && P && (P->n != G->n)) {
    printf("Data validation error: The number of columns for G must be "
           "equal to n.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // c cannot be null.
  if (!c) {
    printf("Data validation error: linear cost term, c, must be provided.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // l + sum(q) should be equal to m.
  QCOSInt sum = l;
  for (QCOSInt i = 0; i < nsoc; ++i) {
    sum += q[i];
  }
  if (G && sum != G->m) {
    printf("Data validation error: l + sum(q) must be equal to m.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // l must be non-negative.
  if (l < 0) {
    printf("Data validation error: l must be non-negative.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  // nsoc must be non-negative.
  if (nsoc < 0) {
    printf("Data validation error: nsoc must be non-negative.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  if ((A && !b) || (b && !A)) {
    printf("Data validation error: If there are equality constraints, A and b "
           "must be provided.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  if ((G && !h) || (h && !G)) {
    printf("Data validation error: If there are conic constraints, G and h "
           "must be provided.");
    return QCOS_DATA_VALIDATION_ERROR;
  }

  return QCOS_NO_ERROR;
}
