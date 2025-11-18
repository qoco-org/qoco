/**
 * @file builtin_linalg.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "builtin_types.h"

QOCOCscMatrix* new_qoco_csc_matrix(const QOCOCscMatrix* A)
{
  QOCOCscMatrix* M = qoco_malloc(sizeof(QOCOCscMatrix));

  if (A) {
    QOCOInt m = A->m;
    QOCOInt n = A->n;
    QOCOInt nnz = A->nnz;

    QOCOFloat* x = qoco_malloc(nnz * sizeof(QOCOFloat));
    QOCOInt* p = qoco_malloc((n + 1) * sizeof(QOCOInt));
    QOCOInt* i = qoco_malloc(nnz * sizeof(QOCOInt));

    copy_arrayf(A->x, x, nnz);
    copy_arrayi(A->i, i, nnz);
    copy_arrayi(A->p, p, n + 1);

    M->m = m;
    M->n = n;
    M->nnz = nnz;
    M->x = x;
    M->i = i;
    M->p = p;
  }
  else {
    M->m = 0;
    M->n = 0;
    M->nnz = 0;
    M->x = NULL;
    M->i = NULL;
    M->p = NULL;
  }

  return M;
}

void free_qoco_csc_matrix(QOCOCscMatrix* A)
{
  free(A->x);
  free(A->i);
  free(A->p);
  free(A);
}

void copy_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

void copy_and_negate_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = -x[i];
  }
}

void copy_arrayi(const QOCOInt* x, QOCOInt* y, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

QOCOFloat qoco_dot(const QOCOFloat* u, const QOCOFloat* v, QOCOInt n)
{
  qoco_assert(u || n == 0);
  qoco_assert(v || n == 0);

  QOCOFloat x = 0.0;
  for (QOCOInt i = 0; i < n; ++i) {
    x += u[i] * v[i];
  }
  return x;
}

QOCOInt max_arrayi(const QOCOInt* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  QOCOInt max = -QOCOInt_MAX;
  for (QOCOInt i = 0; i < n; ++i) {
    max = qoco_max(max, x[i]);
  }
  return max;
}

void scale_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = s * x[i];
  }
}

void qoco_axpy(const QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z,
               QOCOFloat a, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    z[i] = a * x[i] + y[i];
  }
}

void USpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  for (QOCOInt i = 0; i < M->n; i++) {
    r[i] = 0.0;
    for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
      int row = M->i[j];
      r[row] += M->x[j] * v[i];
      if (row != i)
        r[i] += M->x[j] * v[row];
    }
  }
}

void SpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // Clear result buffer.
  for (QOCOInt i = 0; i < M->m; ++i) {
    r[i] = 0.0;
  }

  for (QOCOInt j = 0; j < M->n; j++) {
    for (QOCOInt i = M->p[j]; i < M->p[j + 1]; i++) {
      r[M->i[i]] += M->x[i] * v[j];
    }
  }
}

void SpMtv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r)
{
  qoco_assert(M);
  qoco_assert(v);
  qoco_assert(r);

  // Clear result buffer.
  for (QOCOInt i = 0; i < M->n; ++i) {
    r[i] = 0.0;
  }

  for (QOCOInt i = 0; i < M->n; i++) {
    for (QOCOInt j = M->p[i]; j < M->p[i + 1]; j++) {
      r[i] += M->x[j] * v[M->i[j]];
    }
  }
}

QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  QOCOFloat norm = 0.0;
  QOCOFloat xi;
  for (QOCOInt i = 0; i < n; ++i) {
    xi = qoco_abs(x[i]);
    norm = qoco_max(norm, xi);
  }
  return norm;
}
