/**
 * @file common_linalg.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "common_linalg.h"

void free_qoco_csc_matrix(QOCOCscMatrix* A)
{
  free(A->x);
  free(A->i);
  free(A->p);
  free(A);
}

QOCOCscMatrix* construct_identity_csc(QOCOInt n, QOCOFloat lambda)
{
  QOCOCscMatrix* M = qoco_malloc(sizeof(QOCOCscMatrix));
  QOCOFloat* x = qoco_malloc(n * sizeof(QOCOFloat));
  QOCOInt* p = qoco_malloc((n + 1) * sizeof(QOCOInt));
  QOCOInt* i = qoco_malloc(n * sizeof(QOCOInt));

  M->m = n;
  M->n = n;
  M->nnz = n;
  M->x = x;
  M->i = i;
  M->p = p;

  for (QOCOInt k = 0; k < n; ++k) {
    M->i[k] = k;
    M->x[k] = lambda;
  }

  for (QOCOInt k = 0; k < n + 1; ++k) {
    M->p[k] = k;
  }

  return M;
}

QOCOInt count_diag(QOCOCscMatrix* M)
{
  QOCOInt count = 0;
  for (QOCOInt j = 0; j < M->n; j++) {
    // look for nonempty columns with final element
    // on the diagonal.  Assumes triu format.
    if ((M->p[j + 1] != M->p[j]) && (M->i[M->p[j + 1] - 1] == j)) {
      count++;
    }
  }
  return count;
}

QOCOCscMatrix* regularize_P_csc(QOCOInt num_diagP, QOCOCscMatrix* P,
                                QOCOFloat reg, QOCOInt* nzadded_idx)
{
  QOCOInt n = P->n;

  QOCOInt orig_nnz = P->nnz;
  QOCOInt new_nnz =
      orig_nnz + (n - num_diagP); // We insert one element per missing diagonal

  // Allocate new matrix
  QOCOCscMatrix* Preg = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  Preg->m = n;
  Preg->n = n;
  Preg->nnz = new_nnz;
  Preg->x = (QOCOFloat*)malloc(new_nnz * sizeof(QOCOFloat));
  Preg->i = (QOCOInt*)malloc(new_nnz * sizeof(QOCOInt));
  Preg->p = (QOCOInt*)malloc((n + 1) * sizeof(QOCOInt));

  QOCOInt orig_pos, new_pos = 0;
  Preg->p[0] = 0;

  QOCOInt num_nz = 0;
  for (QOCOInt j = 0; j < n; ++j) {
    QOCOInt inserted_diag = 0;

    for (orig_pos = P->p[j]; orig_pos < P->p[j + 1]; ++orig_pos) {
      QOCOInt row = P->i[orig_pos];

      if (!inserted_diag && row > j) {
        // Insert missing diagonal before this row
        Preg->i[new_pos] = j;
        Preg->x[new_pos] = reg;
        ++new_pos;
        inserted_diag = 1;
      }

      Preg->i[new_pos] = row;
      Preg->x[new_pos] = (row == j) ? P->x[orig_pos] + reg : P->x[orig_pos];
      if (row == j)
        inserted_diag = 1;
      ++new_pos;
    }

    // No entries or diagonal still missing
    if (!inserted_diag) {
      Preg->i[new_pos] = j;
      Preg->x[new_pos] = reg;
      if (nzadded_idx) {
        nzadded_idx[num_nz] = new_pos;
      }
      num_nz++;
      ++new_pos;
    }

    Preg->p[j + 1] = new_pos;
  }
  free_qoco_csc_matrix(P);
  return Preg;
}

void unregularize_csc(QOCOCscMatrix* M, QOCOFloat lambda)
{
  // Iterate over each column.
  for (QOCOInt col = 0; col < M->n; col++) {
    QOCOInt start = M->p[col];
    QOCOInt end = M->p[col + 1];

    // Iterate over the elements in the current column.
    unsigned char insert_set = 0;
    for (QOCOInt i = start; i < end; i++) {
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

void col_inf_norm_USymm(const QOCOCscMatrix* M, QOCOFloat* norm)
{
  for (QOCOInt j = 0; j < M->n; j++) {
    for (QOCOInt idx = M->p[j]; idx < M->p[j + 1]; idx++) {
      QOCOInt row = M->i[idx];
      QOCOFloat val = qoco_abs(M->x[idx]);

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

void row_inf_norm(const QOCOCscMatrix* M, QOCOFloat* norm)
{
  for (QOCOInt i = 0; i < M->m; ++i) {
    QOCOFloat nrm = 0;
    QOCOFloat xi;
    for (QOCOInt nz = 0; nz < M->nnz; ++nz) {
      if (i == M->i[nz]) {
        xi = qoco_abs(M->x[nz]);
        nrm = qoco_max(nrm, xi);
      }
    }
    norm[i] = nrm;
  }
}

QOCOCscMatrix* create_transposed_csc_matrix(const QOCOCscMatrix* A,
                                            QOCOInt* AtoAt)
{
  QOCOCscMatrix* B = qoco_malloc(sizeof(QOCOCscMatrix));
  B->m = A->n;
  B->n = A->m;
  B->nnz = A->nnz;

  // Allocate memory for the transpose matrix.
  B->p = (QOCOInt*)qoco_malloc((A->m + 1) * sizeof(int));
  B->i = (QOCOInt*)qoco_malloc(A->nnz * sizeof(QOCOInt));
  B->x = (double*)qoco_malloc(A->nnz * sizeof(QOCOFloat));

  // Count the number of non-zeros in each row.
  QOCOInt* row_counts = (QOCOInt*)calloc(A->m, sizeof(QOCOInt));
  for (int j = 0; j < A->n; j++) {
    for (int i = A->p[j]; i < A->p[j + 1]; i++) {
      row_counts[A->i[i]]++;
    }
  }

  B->p[0] = 0;
  for (int i = 0; i < A->m; i++) {
    B->p[i + 1] = B->p[i] + row_counts[i];
  }

  QOCOInt* temp = (int*)calloc(
      A->m, sizeof(int)); // To track the insertion position for each row
  for (int j = 0; j < A->n; j++) {
    for (int i = A->p[j]; i < A->p[j + 1]; i++) {
      int row = A->i[i];
      int dest_pos = B->p[row] + temp[row];
      B->i[dest_pos] = j;       // Column index becomes row index
      B->x[dest_pos] = A->x[i]; // Value remains the same
      if (AtoAt) {
        AtoAt[i] = dest_pos;
      }
      temp[row]++;
    }
  }

  // Clean up
  qoco_free(row_counts);
  qoco_free(temp);

  return B;
}

void row_col_scale_csc(const QOCOCscMatrix* M, QOCOFloat* E, QOCOFloat* D)
{
  for (QOCOInt j = 0; j < M->n; ++j) {
    for (QOCOInt i = M->p[j]; i < M->p[j + 1]; ++i) {
      M->x[i] *= (D[j] * E[M->i[i]]);
    }
  }
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

QOCOInt max_arrayi(const QOCOInt* x, QOCOInt n)
{
  qoco_assert(x || n == 0);

  QOCOInt max = -QOCOInt_MAX;
  for (QOCOInt i = 0; i < n; ++i) {
    max = qoco_max(max, x[i]);
  }
  return max;
}

void scale_arrayf(QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n)
{
  qoco_assert(x || n == 0);
  qoco_assert(y || n == 0);

  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = s * x[i];
  }
}

void ew_product_arrayf(QOCOFloat* x, QOCOFloat* y, QOCOFloat* z, QOCOInt n)
{
  for (QOCOInt i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

void invert_permutation(const QOCOInt* p, QOCOInt* pinv, QOCOInt n)
{
  for (QOCOInt i = 0; i < n; ++i) {
    pinv[p[i]] = i;
  }
}

QOCOInt cumsum(QOCOInt* p, QOCOInt* c, QOCOInt n)
{
  qoco_assert(p);
  qoco_assert(c);

  QOCOInt nz = 0;
  for (QOCOInt i = 0; i < n; i++) {
    p[i] = nz;
    nz += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz;
}

QOCOCscMatrix* csc_symperm(const QOCOCscMatrix* A, const QOCOInt* pinv,
                           QOCOInt* AtoC)
{
  QOCOInt i, j, p, q, i2, j2, n;
  QOCOInt* Ap;
  QOCOInt* Ai;
  QOCOInt* Cp;
  QOCOInt* Ci;
  QOCOInt* w;
  QOCOFloat* Cx;
  QOCOFloat* Ax;
  QOCOCscMatrix* C;

  n = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C = new_qoco_csc_matrix(A);
  w = qoco_calloc(n, sizeof(QOCOInt));

  qoco_assert(C);
  qoco_assert(w);

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
      w[qoco_max(i2, j2)]++;   /* column count of C */
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
      Ci[q = w[qoco_max(i2, j2)]++] = qoco_min(i2, j2);

      if (Cx)
        Cx[q] = Ax[p];

      AtoC[p] = q;
    }
  }
  qoco_free(w);
  return C;
}
