#include "linalg.h"

QCOSVector* qcos_vector_calloc(QCOSInt n)
{
  QCOSVector* v = qcos_malloc(sizeof(QCOSVector));
  v->n = n;
  v->x = qcos_calloc(n, sizeof(QCOSFloat));
  return v;
}

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

QCOSVector* new_qcos_vector_from_array(QCOSFloat* x, QCOSInt n)
{
  QCOSVector* v = qcos_malloc(sizeof(QCOSVector));
  QCOSFloat* y = qcos_malloc(n * sizeof(QCOSFloat));

  copy_arrayf(x, y, n);
  v->n = n;
  v->x = y;
  return v;
}

void copy_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n)
{
  for (QCOSInt i = 0; i < n; ++i) {
    y[i] = x[i];
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