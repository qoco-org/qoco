#include "test_utils.h"

void expect_eq_vectorf(QOCOFloat* x, QOCOFloat* y, QOCOInt n, QOCOFloat tol)
{
  for (QOCOInt i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], y[i], tol);
  }
}

void expect_eq_csc(QOCOCscMatrix* A, QOCOCscMatrix* B, QOCOFloat tol)
{
  EXPECT_EQ(A->m, B->m);
  EXPECT_EQ(A->n, B->n);
  EXPECT_EQ(A->nnz, B->nnz);

  for (QOCOInt i = 0; i < A->nnz; ++i) {
    EXPECT_EQ(A->i[i], B->i[i]);
  }

  for (QOCOInt i = 0; i < A->n + 1; ++i) {
    EXPECT_EQ(A->p[i], B->p[i]);
  }

  expect_eq_vectorf(A->x, B->x, A->nnz, tol);
}

void expect_rel_error(QOCOFloat x, QOCOFloat y, QOCOFloat tol)
{
  QOCOFloat err = x - y;
  err = qoco_abs(err);
  QOCOFloat yabs = qoco_abs(y);
  err = err / yabs;
  EXPECT_LE(err, tol);
}