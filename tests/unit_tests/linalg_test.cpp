#include "test_utils.h"
#include "gtest/gtest.h"

#include "qcos.h"

TEST(linalg, new_qcos_csc_matrix_test)
{
  QCOSInt m = 5;
  QCOSInt n = 3;
  QCOSFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QCOSInt Annz = 12;
  QCOSInt Ap[] = {0, 4, 8, 12};
  QCOSInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  QCOSCscMatrix* M = new_qcos_csc_matrix(A);

  EXPECT_EQ(A->m, M->m);
  EXPECT_EQ(A->n, M->n);
  EXPECT_EQ(A->nnz, M->nnz);
  for (QCOSInt k = 0; k < Annz; ++k) {
    EXPECT_EQ(A->i[k], M->i[k]);
  }
  for (QCOSInt k = 0; k < n + 1; ++k) {
    EXPECT_EQ(A->p[k], M->p[k]);
  }
  expect_eq_vectorf(A->x, M->x, Annz, tol);

  free(A);
  free(M->x);
  free(M->i);
  free(M->p);
  free(M);
}

TEST(linalg, copy_arrayf_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat y[n];
  QCOSFloat tol = 1e-12;

  copy_arrayf(x, y, n);
  expect_eq_vectorf(x, y, n, tol);
}

TEST(linalg, copy_and_negate_arrayf_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat y[n];
  QCOSFloat tol = 1e-12;

  copy_and_negate_arrayf(x, y, n);
  for (QCOSInt i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], -y[i], tol);
  }
}

TEST(linalg, copy_arrayi_test)
{
  constexpr QCOSInt n = 6;
  QCOSInt x[] = {1, 2, 3, 4, 5, 6};
  QCOSInt y[n];

  copy_arrayi(x, y, n);
  for (QCOSInt i = 0; i < n; ++i) {
    EXPECT_EQ(x[i], y[i]);
  }
}

TEST(linalg, dot_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat y[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  QCOSFloat tol = 1e-12;
  QCOSFloat expected_ans = 217.0;

  EXPECT_NEAR(dot(x, y, n), expected_ans, tol);
}

TEST(linalg, max_arrayi_test)
{
  constexpr QCOSInt n = 6;
  QCOSInt x[] = {1, 2, 3, 9, 5, 6};
  QCOSInt expected_ans = 9;

  EXPECT_EQ(max_arrayi(x, n), expected_ans);
}

TEST(linalg, scale_arrayf_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat yexpected[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QCOSFloat y[n];
  QCOSFloat s = 0.5;
  QCOSFloat tol = 1e-12;

  scale_arrayf(x, y, s, n);
  expect_eq_vectorf(y, yexpected, n, tol);
}

TEST(linalg, axpy_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat y[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QCOSFloat z[n];
  QCOSFloat zexpected[n] = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0};
  QCOSFloat a = 2.0;
  QCOSFloat tol = 1e-12;

  axpy(x, y, z, a, n);
  expect_eq_vectorf(z, zexpected, n, tol);
}

TEST(linalg, USpMv_test)
{
  constexpr QCOSInt m = 5;
  constexpr QCOSInt n = 5;
  QCOSFloat Ax[] = {1, 2, 3, 5, 4, 6, 8, 7};
  QCOSInt Annz = 12;
  QCOSInt Ap[] = {0, 1, 2, 4, 7, 8};
  QCOSInt Ai[] = {0, 0, 0, 1, 0, 1, 3, 1};
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  QCOSFloat v[] = {2.0, 4.0, 6.0, 8.0, 10.0};
  QCOSFloat rexpected[] = {60.0, 152.0, 26.0, 96.0, 28.0};
  QCOSFloat r[n];
  QCOSFloat tol = 1e-12;

  USpMv(A, v, r);
  expect_eq_vectorf(r, rexpected, n, tol);
  free(A);
}