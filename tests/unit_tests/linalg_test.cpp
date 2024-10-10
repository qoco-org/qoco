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
  free_qcos_csc_matrix(M);
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

TEST(linalg, scale_arrayf_inplace_test)
{
  constexpr QCOSInt n = 6;
  QCOSFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QCOSFloat xexpected[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QCOSFloat s = 0.5;
  QCOSFloat tol = 1e-12;

  scale_arrayf(x, x, s, n);
  expect_eq_vectorf(x, xexpected, n, tol);
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

TEST(linalg, SpMv_test)
{
  constexpr QCOSInt m = 5;
  constexpr QCOSInt n = 3;
  QCOSFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QCOSInt Annz = 12;
  QCOSInt Ap[] = {0, 4, 8, 12};
  QCOSInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QCOSFloat v[] = {1.0, 2.0, 3.0};
  QCOSFloat r[m];
  QCOSFloat rexpected[] = {14.0, 22.0, 43.0, 32.0, 26.0};

  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  SpMv(A, v, r);

  expect_eq_vectorf(r, rexpected, n, tol);

  free(A);
}

TEST(linalg, SpMtv_test)
{
  constexpr QCOSInt m = 5;
  constexpr QCOSInt n = 3;
  QCOSFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QCOSInt Annz = 12;
  QCOSInt Ap[] = {0, 4, 8, 12};
  QCOSInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QCOSFloat v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  QCOSFloat r[n];
  QCOSFloat rexpected[] = {64.0, 90.0, 67.0};

  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  SpMtv(A, v, r);

  expect_eq_vectorf(r, rexpected, n, tol);

  free(A);
}

TEST(linalg, inf_norm_test)
{
  constexpr QCOSInt n = 4;
  QCOSFloat x[] = {-1.5, 6.0, -10.0, 8.0};
  QCOSFloat expected_ans = 10.0;
  QCOSFloat tol = 1e-12;

  EXPECT_NEAR(inf_norm(x, n), expected_ans, tol);
}

TEST(linalg, regularize_test1)
{
  QCOSInt n = 6;
  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat Px_exp[] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* Pexp = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(Pexp, n, n, Pnnz, Px_exp, Pp, Pi);

  regularize(P, 0.5, NULL);
  expect_eq_csc(P, Pexp, tol);

  free(P);
  free(Pexp);
}

TEST(linalg, regularize_test2)
{
  constexpr QCOSInt n = 3;
  QCOSFloat Px[] = {1, 2, 2, 3, 3};
  constexpr QCOSInt Pnnz = 5;
  QCOSInt Pp[] = {0, 2, 5, 5};
  QCOSInt Pi[] = {1, 2, 0, 1, 2};

  QCOSFloat Px_exp[] = {1, 1, 2, 2, 4, 3, 1};
  QCOSInt Pnnz_exp = 7;
  QCOSInt Pp_exp[] = {0, 3, 6, 7};
  QCOSInt Pi_exp[] = {0, 1, 2, 0, 1, 2, 2};

  QCOSInt nzadded_idx[n];
  QCOSInt nzadded_idx_exp[] = {0, 6};
  QCOSInt nz_added_exp = 2;

  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* Pexp = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(Pexp, n, n, Pnnz_exp, Px_exp, Pp_exp, Pi_exp);

  QCOSCscMatrix* Pmalloc = new_qcos_csc_matrix(P);
  QCOSCscMatrix* Pexpmalloc = new_qcos_csc_matrix(Pexp);

  QCOSInt nz_added = regularize(Pmalloc, 1.0, nzadded_idx);

  expect_eq_csc(Pmalloc, Pexpmalloc, tol);

  EXPECT_EQ(nz_added, nz_added_exp);

  for (QCOSInt i = 0; i < nz_added; ++i) {
    EXPECT_EQ(nzadded_idx[i], nzadded_idx_exp[i]);
  }

  free(P);
  free(Pexp);
  free_qcos_csc_matrix(Pmalloc);
  free_qcos_csc_matrix(Pexpmalloc);
}

TEST(linalg, regularize_test3)
{
  QCOSInt n = 6;
  QCOSFloat Px[] = {1, 2, 3, 3, 5, 7, 8};
  QCOSInt Pnnz = 7;
  QCOSInt Pp[] = {0, 1, 3, 4, 5, 7, 7};
  QCOSInt Pi[] = {0, 0, 1, 0, 2, 2, 4};

  QCOSFloat Pxexp[] = {1.001, 2.0,   3.001, 3.0,   0.001,
                       5.0,   0.001, 7.0,   8.001, 0.001};
  QCOSInt Pnnzexp = 10;
  QCOSInt Ppexp[] = {0, 1, 3, 5, 7, 9, 10};
  QCOSInt Piexp[] = {0, 0, 1, 0, 2, 2, 3, 2, 4, 5};
  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* Pexp = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(Pexp, n, n, Pnnzexp, Pxexp, Ppexp, Piexp);

  QCOSCscMatrix* Pmalloc = new_qcos_csc_matrix(P);

  regularize(Pmalloc, 1e-3, NULL);
  expect_eq_csc(Pmalloc, Pexp, tol);

  free_qcos_csc_matrix(Pmalloc);
  free(P);
  free(Pexp);
}

TEST(linalg, col_inf_norm_USymm_test)
{
  constexpr QCOSInt m = 3;
  constexpr QCOSInt n = 3;
  QCOSFloat Ax[] = {1, 2, 5, 3, 6, 8};
  QCOSInt Annz = 6;
  QCOSInt Ap[] = {0, 1, 3, 6};
  QCOSInt Ai[] = {0, 0, 1, 0, 1, 2};
  QCOSFloat norm[n];
  QCOSFloat norm_expected[] = {3, 6, 8};

  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  col_inf_norm_USymm(A, norm);

  expect_eq_vectorf(norm, norm_expected, n, tol);

  free(A);
}

TEST(linalg, row_inf_norm_test)
{
  constexpr QCOSInt m = 5;
  constexpr QCOSInt n = 3;
  QCOSFloat Ax[] = {1, 4, 10, 3, 2, 8, -11, 4, 3, -6, 9, 5};
  QCOSInt Annz = 12;
  QCOSInt Ap[] = {0, 4, 8, 12};
  QCOSInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QCOSFloat norm[m];
  QCOSFloat norm_expected[] = {3.0, 6.0, 9.0, 11.0, 5.0};

  QCOSFloat tol = 1e-12;

  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  row_inf_norm(A, norm);

  expect_eq_vectorf(norm, norm_expected, n, tol);

  free(A);
}

TEST(linalg, ruiz_test)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 3;
  QCOSInt nsoc = 1;

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat Ax[] = {1, 1, 1, 2};
  QCOSInt Annz = 4;
  QCOSInt Ap[] = {0, 1, 3, 4, 4, 4, 4};
  QCOSInt Ai[] = {0, 0, 1, 1};

  QCOSFloat Gx[] = {-1, -1, -1, -1, -1, -1};
  QCOSInt Gnnz = 6;
  QCOSInt Gp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Gi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat c[] = {1, 2, 3, 4, 5, 6};
  QCOSFloat b[] = {1, 2};
  QCOSFloat h[] = {0, 0, 0, 0, 0, 0};
  QCOSInt q[] = {3};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat Dexp[] = {1.0000, 0.8409, 0.6389, 0.7071, 0.7022, 0.6894};
  QCOSFloat Eexp[] = {1.0000, 0.7825};
  QCOSFloat Fexp[] = {1.0000, 1.1892, 1.5315, 1.4142, 1.4142, 1.4142};
  QCOSFloat kexp = 0.2480;
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->ruiz_iters = 5;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

  expect_eq_vectorf(solver->work->kkt->Druiz, Dexp, n, tol);
  expect_eq_vectorf(solver->work->kkt->Eruiz, Eexp, p, tol);
  expect_eq_vectorf(solver->work->kkt->Fruiz, Fexp, m, tol);
  EXPECT_NEAR(solver->work->kkt->k, kexp, tol);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}
