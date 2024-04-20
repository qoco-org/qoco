#include "linalg.h"

QCOSCscMatrix* new_qcos_csc_matrix(QCOSCscMatrix* A)
{
  QCOSInt m = A->m;
  QCOSInt n = A->n;
  QCOSInt nnz = A->nnz;

  QCOSCscMatrix* M = qcos_malloc(sizeof(QCOSCscMatrix));
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

  return M;
}

void copy_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

void copy_and_negate_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = -x[i];
  }
}

void copy_arrayi(const QCOSInt* x, QCOSInt* y, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

QCOSFloat dot(QCOSFloat* u, QCOSFloat* v, QCOSInt n)
{
  QCOSFloat x = 0.0;
  for (QCOSInt i = 0; i < n; ++i) {
    x += u[i] * v[i];
  }
  return x;
}

void qcos_USpMv(QCOSCscMatrix* M, QCOSFloat* v, QCOSFloat* r)
{
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