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

QCOSCscMatrix* new_qcos_csc_matrix(const QCOSCscMatrix* A)
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

QCOSCscMatrix* construct_identity(QCOSInt n, QCOSFloat lambda)
{
  QCOSCscMatrix* M = qcos_malloc(sizeof(QCOSCscMatrix));
  QCOSFloat* x = qcos_malloc(n * sizeof(QCOSFloat));
  QCOSInt* p = qcos_malloc((n + 1) * sizeof(QCOSInt));
  QCOSInt* i = qcos_malloc(n * sizeof(QCOSInt));

  M->m = n;
  M->n = n;
  M->nnz = n;
  M->x = x;
  M->i = i;
  M->p = p;

  for (QCOSInt k = 0; k < n; ++k) {
    M->i[k] = k;
    M->x[k] = lambda;
  }

  for (QCOSInt k = 0; k < n + 1; ++k) {
    M->p[k] = k;
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

QCOSFloat inf_norm(const QCOSFloat* x, QCOSInt n)
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

QCOSInt regularize(QCOSCscMatrix* M, QCOSFloat lambda, QCOSInt* nzadded_idx)
{
  QCOSInt num_nz = 0;
  // Iterate over each column.
  for (QCOSInt col = 0; col < M->n; col++) {
    QCOSInt start = M->p[col];
    QCOSInt end = M->p[col + 1];

    // Flag to check if the diagonal element exists.
    QCOSInt diagonal_exists = 0;

    // Iterate over the elements in the current column.
    unsigned char insert_set = 0;
    QCOSInt insert = end;
    for (QCOSInt i = start; i < end; i++) {
      if (!insert_set && M->i[i] > col) {
        insert = i;
        insert_set = 1;
      }
      if (M->i[i] == col) {
        M->x[i] += lambda; // Add lambda to the diagonal element.
        diagonal_exists = 1;
        break;
      }
    }

    // If the diagonal element does not exist, we need to insert it.
    if (!diagonal_exists) {
      // Shift all the elements in values and row_indices arrays to make space
      // for the new diagonal element.
      M->nnz++;
      M->x = realloc(M->x, M->nnz * sizeof(QCOSFloat));
      M->i = realloc(M->i, M->nnz * sizeof(QCOSInt));

      for (QCOSInt i = M->nnz - 1; i > insert; i--) {
        M->x[i] = M->x[i - 1];
        M->i[i] = M->i[i - 1];
      }

      // Insert the new diagonal element.
      M->x[insert] = lambda;
      M->i[insert] = col;
      if (nzadded_idx) {
        nzadded_idx[num_nz] = insert;
      }
      num_nz++;

      // Update the column_pointers array.
      for (QCOSInt i = col + 1; i <= M->n; i++) {
        M->p[i]++;
      }
    }
  }
  return num_nz;
}

void unregularize(QCOSCscMatrix* M, QCOSFloat lambda)
{
  // Iterate over each column.
  for (QCOSInt col = 0; col < M->n; col++) {
    QCOSInt start = M->p[col];
    QCOSInt end = M->p[col + 1];

    // Iterate over the elements in the current column.
    unsigned char insert_set = 0;
    for (QCOSInt i = start; i < end; i++) {
      if (!insert_set && M->i[i] > col) {
        insert_set = 1;
      }
      if (M->i[i] == col) {
        M->x[i] -= lambda; // Add lambda to the diagonal element.
        break;
      }
    }
  }
}

void col_inf_norm_USymm(const QCOSCscMatrix* M, QCOSFloat* norm)
{
  for (QCOSInt j = 0; j < M->n; j++) {
    for (QCOSInt idx = M->p[j]; idx < M->p[j + 1]; idx++) {
      QCOSInt row = M->i[idx];
      QCOSFloat val = qcos_abs(M->x[idx]);

      if (val > norm[j]) {
        norm[j] = val;
      }

      if (row != j) {
        if (val > norm[row]) {
          norm[row] = val;
        }
      }
    }
  }
}

void row_inf_norm(const QCOSCscMatrix* M, QCOSFloat* norm)
{
  for (QCOSInt i = 0; i < M->m; ++i) {
    QCOSFloat nrm = 0;
    QCOSFloat xi;
    for (QCOSInt nz = 0; nz < M->nnz; ++nz) {
      if (i == M->i[nz]) {
        xi = qcos_abs(M->x[nz]);
        nrm = qcos_max(nrm, xi);
      }
    }
    norm[i] = nrm;
  }
}

void col_scale(const QCOSCscMatrix* M, QCOSFloat* S)
{
  for (QCOSInt j = 0; j < M->n; ++j) {
    for (QCOSInt i = M->p[j]; i < M->p[j + 1]; ++i) {
      M->x[i] *= S[j];
    }
  }
}

void row_scale(const QCOSCscMatrix* M, QCOSFloat* S)
{
  for (QCOSInt i = 0; i < M->m; ++i) {
    for (QCOSInt nz = 0; nz < M->nnz; ++nz) {
      if (i == M->i[nz]) {
        M->x[nz] *= S[i];
      }
    }
  }
}

void ew_product(QCOSFloat* x, const QCOSFloat* y, QCOSFloat* z, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

void invert_permutation(const QCOSInt* p, QCOSInt* pinv, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    pinv[p[i]] = i;
  }
}

QCOSInt cumsum(QCOSInt* p, QCOSInt* c, QCOSInt n)
{
  qcos_assert(p);
  qcos_assert(c);

  QCOSInt nz = 0;
  for (QCOSInt i = 0; i < n; i++) {
    p[i] = nz;
    nz += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz;
}

QCOSCscMatrix* csc_symperm(const QCOSCscMatrix* A, const QCOSInt* pinv,
                           QCOSInt* AtoC)
{
  QCOSInt i, j, p, q, i2, j2, n;
  QCOSInt* Ap;
  QCOSInt* Ai;
  QCOSInt* Cp;
  QCOSInt* Ci;
  QCOSInt* w;
  QCOSFloat* Cx;
  QCOSFloat* Ax;
  QCOSCscMatrix* C;

  n = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C = new_qcos_csc_matrix(A);
  w = qcos_calloc(n, sizeof(QCOSInt));

  qcos_assert(C);
  qcos_assert(w);

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (j = 0; j < n; j++) /* count entries in each column of C */
  {
    j2 = pinv ? pinv[j] : j; /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j)
        continue;              /* skip lower triangular part of A */
      i2 = pinv ? pinv[i] : i; /* row i of A is row i2 of C */
      w[qcos_max(i2, j2)]++;   /* column count of C */
    }
  }
  cumsum(Cp, w, n); /* compute column pointers of C */

  for (j = 0; j < n; j++) {
    j2 = pinv ? pinv[j] : j; /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j)
        continue;              /* skip lower triangular
                                  part of A*/
      i2 = pinv ? pinv[i] : i; /* row i of A is row i2
                                  of C */
      Ci[q = w[qcos_max(i2, j2)]++] = qcos_min(i2, j2);

      if (Cx)
        Cx[q] = Ax[p];

      AtoC[p] = q;
    }
  }
  qcos_free(w);
  return C;
}
