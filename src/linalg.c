/**
 * @file linalg.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "linalg.h"

QCOSCscMatrix* new_qcos_csc_matrix(QCOSCscMatrix* A)
{
  QCOSCscMatrix* M = qcos_malloc(sizeof(QCOSCscMatrix));

  if (A) {
    QCOSInt m = A->m;
    QCOSInt n = A->n;
    QCOSInt nnz = A->nnz;

    QCOSFloat* x = qcos_malloc(nnz * sizeof(QCOSFloat));
    QCOSInt* p = qcos_malloc((n + 1) * sizeof(QCOSInt));
    QCOSInt* i = qcos_malloc(nnz * sizeof(QCOSInt));

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

void free_qcos_csc_matrix(QCOSCscMatrix* A)
{
  free(A->x);
  free(A->i);
  free(A->p);
  free(A);
}

void copy_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n)
{
  qcos_assert(x || n == 0);
  qcos_assert(y || n == 0);

  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

void copy_and_negate_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n)
{
  qcos_assert(x || n == 0);
  qcos_assert(y || n == 0);

  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = -x[i];
  }
}

void copy_arrayi(const QCOSInt* x, QCOSInt* y, QCOSInt n)
{
  qcos_assert(x || n == 0);
  qcos_assert(y || n == 0);

  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

QCOSFloat dot(const QCOSFloat* u, const QCOSFloat* v, QCOSInt n)
{
  qcos_assert(u || n == 0);
  qcos_assert(v || n == 0);

  QCOSFloat x = 0.0;
  for (QCOSInt i = 0; i < n; ++i) {
    x += u[i] * v[i];
  }
  return x;
}

QCOSInt max_arrayi(const QCOSInt* x, QCOSInt n)
{
  qcos_assert(x || n == 0);

  QCOSInt max = -QCOSInt_MAX;
  for (QCOSInt i = 0; i < n; ++i) {
    max = qcos_max(max, x[i]);
  }
  return max;
}

void scale_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSFloat s, QCOSInt n)
{
  qcos_assert(x || n == 0);
  qcos_assert(y || n == 0);

  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = s * x[i];
  }
}

void axpy(const QCOSFloat* x, const QCOSFloat* y, QCOSFloat* z, QCOSFloat a,
          QCOSInt n)
{
  qcos_assert(x || n == 0);
  qcos_assert(y || n == 0);

  for (QCOSInt i = 0; i < n; ++i) {
    z[i] = a * x[i] + y[i];
  }
}

void USpMv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r)
{
  qcos_assert(M);
  qcos_assert(v);
  qcos_assert(r);

  for (QCOSInt i = 0; i < M->n; i++) {
    r[i] = 0.0;
    for (QCOSInt j = M->p[i]; j < M->p[i + 1]; j++) {
      int row = M->i[j];
      r[row] += M->x[j] * v[i];
      if (row != i)
        r[i] += M->x[j] * v[row];
    }
  }
}

void SpMv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r)
{
  qcos_assert(M);
  qcos_assert(v);
  qcos_assert(r);

  // Clear result buffer.
  for (QCOSInt i = 0; i < M->m; ++i) {
    r[i] = 0.0;
  }

  for (QCOSInt j = 0; j < M->n; j++) {
    for (QCOSInt i = M->p[j]; i < M->p[j + 1]; i++) {
      r[M->i[i]] += M->x[i] * v[j];
    }
  }
}

void SpMtv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r)
{
  qcos_assert(M);
  qcos_assert(v);
  qcos_assert(r);

  // Clear result buffer.
  for (QCOSInt i = 0; i < M->n; ++i) {
    r[i] = 0.0;
  }

  for (QCOSInt i = 0; i < M->n; i++) {
    for (QCOSInt j = M->p[i]; j < M->p[i + 1]; j++) {
      r[i] += M->x[j] * v[M->i[j]];
    }
  }
}

QCOSFloat norm_inf(const QCOSFloat* x, QCOSInt n)
{
  qcos_assert(x || n == 0);

  QCOSFloat norm = 0.0;
  QCOSFloat xi;
  for (QCOSInt i = 0; i < n; ++i) {
    xi = qcos_abs(x[i]);
    norm = qcos_max(norm, xi);
  }
  return norm;
}