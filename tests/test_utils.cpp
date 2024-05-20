#include "test_utils.h"

void expect_eq_vectorf(QCOSFloat* x, QCOSFloat* y, QCOSInt n, QCOSFloat tol)
{
  for (QCOSInt i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], y[i], tol);
  }
}

void expect_eq_csc(QCOSCscMatrix* A, QCOSCscMatrix* B, QCOSFloat tol)
{
  EXPECT_EQ(A->m, B->m);
  EXPECT_EQ(A->n, B->n);
  EXPECT_EQ(A->nnz, B->nnz);

  for (QCOSInt i = 0; i < A->nnz; ++i) {
    EXPECT_EQ(A->i[i], B->i[i]);
  }

  for (QCOSInt i = 0; i < A->n + 1; ++i) {
    EXPECT_EQ(A->p[i], B->p[i]);
  }

  expect_eq_vectorf(A->x, B->x, A->nnz, tol);
}