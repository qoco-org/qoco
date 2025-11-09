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

QOCOMatrix* new_qoco_matrix(const QOCOCscMatrix* A)
{
  QOCOMatrix* M = qoco_malloc(sizeof(QOCOMatrix));
  QOCOCscMatrix* Mcsc = qoco_malloc(sizeof(QOCOCscMatrix));

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

    Mcsc->m = m;
    Mcsc->n = n;
    Mcsc->nnz = nnz;
    Mcsc->x = x;
    Mcsc->i = i;
    Mcsc->p = p;
  }
  else {
    Mcsc->m = 0;
    Mcsc->n = 0;
    Mcsc->nnz = 0;
    Mcsc->x = NULL;
    Mcsc->i = NULL;
    Mcsc->p = NULL;
  }

  M->csc = M;

  return M;
}

QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n)
{
  QOCOVectorf* v = qoco_malloc(sizeof(QOCOVectorf));
  QOCOFloat* vdata = qoco_malloc(sizeof(QOCOFloat) * n);
  copy_arrayf(x, vdata, n);

  v->len = n;
  v->data = vdata;
}

void free_qoco_matrix(QOCOMatrix* A)
{
  free_qoco_csc_matrix(A);
  qoco_free(A);
}

void free_qoco_vectorf(QOCOVectorf* x)
{
  qoco_free(x->data);
  qoco_free(x);
}

QOCOMatrix* create_transposed_matrix(const QOCOMatrix* A, QOCOInt* AtoAt)
{
  QOCOMatrix* At = qoco_malloc(sizeof(QOCOMatrix));
  QOCOCscMatrix* Atcsc = create_transposed_csc_matrix(A->csc, AtoAt);
  At->csc = Atcsc;
  return At;
}

QOCOMatrix* regularize_P(QOCOInt num_diagP, QOCOMatrix* P, QOCOFloat reg,
                         QOCOInt* nzadded_idx)
{
  regularize_P_csc(num_diagP, P->csc, reg, nzadded_idx);
  qoco_free(P);
}

void unregularize(QOCOMatrix* M, QOCOFloat lambda)
{
  unregularize_csc(M->csc, lambda);
}

QOCOMatrix* construct_identity(QOCOInt n, QOCOFloat lambda)
{
  QOCOMatrix* M = qoco_malloc(sizeof(QOCOMatrix));
  M->csc = construct_identity_csc(n, lambda);
}

void scale_matrix(QOCOFloat a, QOCOMatrix* M)
{
  scale_arrayf(M->csc->x, M->csc->x, a, M->csc->nnz);
}

void row_col_scale(QOCOMatrix* M, QOCOVectorf* E, QOCOVectorf* D)
{
  row_col_scale_csc(M->csc, E->data, D->data);
}

void update_matrix(QOCOMatrix* M, QOCOFloat* Mnew)
{
  copy_arrayf(Mnew, M->csc->x, M->csc->nnz);
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

void ew_product(QOCOVectorf* x, QOCOVectorf* y, QOCOVectorf* z)
{
  qoco_assert(x->len == y->len);
  qoco_assert(x->len == z->len);
  ew_product_arrayf(x->data, y->data, z->data, x->len);
}

void ew_product_vec_array(QOCOVectorf* x, QOCOFloat* y, QOCOVectorf* z)
{
  qoco_assert(x->len == y->len);
  qoco_assert(x->len == z->len);
  ew_product_arrayf(x->data, y, z->data, x->len);
}

void scale_vectorf(QOCOFloat a, QOCOVectorf* u)
{
  scale_arrayf(u->data, u->data, a, u->len);
}

void copy_vectorf(QOCOVectorf* src, QOCOFloat* dest, QOCOInt dest_idx,
                  QOCOInt negate)
{
  if (negate) {
    copy_arrayf(src->data, &dest[dest_idx], src->len);
  }
  else {
    copy_and_negate_arrayf(src->data, &dest[dest_idx], src->len);
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
