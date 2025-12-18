#include "qoco.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(linalg, new_qoco_csc_matrix_test)
{
  QOCOInt m = 5;
  QOCOInt n = 3;
  QOCOFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QOCOInt Annz = 12;
  QOCOInt Ap[] = {0, 4, 8, 12};
  QOCOInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  QOCOCscMatrix* M = new_qoco_csc_matrix(A);

  EXPECT_EQ(A->m, M->m);
  EXPECT_EQ(A->n, M->n);
  EXPECT_EQ(A->nnz, M->nnz);
  for (QOCOInt k = 0; k < Annz; ++k) {
    EXPECT_EQ(A->i[k], M->i[k]);
  }
  for (QOCOInt k = 0; k < n + 1; ++k) {
    EXPECT_EQ(A->p[k], M->p[k]);
  }
  expect_eq_vectorf(A->x, M->x, Annz, tol);

  free(A);
  free_qoco_csc_matrix(M);
}

TEST(linalg, copy_arrayf_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat y[n];
  QOCOFloat tol = 1e-12;

  copy_arrayf(x, y, n);
  expect_eq_vectorf(x, y, n, tol);
}

TEST(linalg, copy_and_negate_arrayf_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat y[n];
  QOCOFloat tol = 1e-12;

  copy_and_negate_arrayf(x, y, n);
  for (QOCOInt i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], -y[i], tol);
  }
}

TEST(linalg, copy_arrayi_test)
{
  constexpr QOCOInt n = 6;
  QOCOInt x[] = {1, 2, 3, 4, 5, 6};
  QOCOInt y[n];

  copy_arrayi(x, y, n);
  for (QOCOInt i = 0; i < n; ++i) {
    EXPECT_EQ(x[i], y[i]);
  }
}

TEST(linalg, qoco_dot_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat y[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  QOCOFloat tol = 1e-12;
  QOCOFloat expected_ans = 217.0;

  EXPECT_NEAR(qoco_dot(x, y, n), expected_ans, tol);
}

TEST(linalg, max_arrayi_test)
{
  constexpr QOCOInt n = 6;
  QOCOInt x[] = {1, 2, 3, 9, 5, 6};
  QOCOInt expected_ans = 9;

  EXPECT_EQ(max_arrayi(x, n), expected_ans);
}

TEST(linalg, scale_arrayf_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat yexpected[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QOCOFloat y[n];
  QOCOFloat s = 0.5;
  QOCOFloat tol = 1e-12;

  scale_arrayf(x, y, s, n);
  expect_eq_vectorf(y, yexpected, n, tol);
}

TEST(linalg, scale_arrayf_inplace_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat xexpected[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QOCOFloat s = 0.5;
  QOCOFloat tol = 1e-12;

  scale_arrayf(x, x, s, n);
  expect_eq_vectorf(x, xexpected, n, tol);
}

TEST(linalg, qoco_axpy_test)
{
  constexpr QOCOInt n = 6;
  QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  QOCOFloat y[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  QOCOFloat z[n];
  QOCOFloat zexpected[n] = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0};
  QOCOFloat a = 2.0;
  QOCOFloat tol = 1e-12;

  qoco_axpy(x, y, z, a, n);
  expect_eq_vectorf(z, zexpected, n, tol);
}

TEST(linalg, USpMv_test)
{
  constexpr QOCOInt m = 5;
  constexpr QOCOInt n = 5;
  QOCOFloat Ax[] = {1, 2, 3, 5, 4, 6, 8, 7};
  QOCOInt Annz = 12;
  QOCOInt Ap[] = {0, 1, 2, 4, 7, 8};
  QOCOInt Ai[] = {0, 0, 0, 1, 0, 1, 3, 1};
  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);
  QOCOMatrix* Amat = new_qoco_matrix(A);

  QOCOFloat v[] = {2.0, 4.0, 6.0, 8.0, 10.0};
  QOCOFloat rexpected[] = {60.0, 152.0, 26.0, 96.0, 28.0};
  QOCOFloat r[n];
  QOCOFloat tol = 1e-12;

  USpMv(Amat, v, r);
  expect_eq_vectorf(r, rexpected, n, tol);
  free(A);
  free_qoco_matrix(Amat);
}

TEST(linalg, SpMv_test)
{
  constexpr QOCOInt m = 5;
  constexpr QOCOInt n = 3;
  QOCOFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QOCOInt Annz = 12;
  QOCOInt Ap[] = {0, 4, 8, 12};
  QOCOInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QOCOFloat v[] = {1.0, 2.0, 3.0};
  QOCOFloat r[m];
  QOCOFloat rexpected[] = {14.0, 22.0, 43.0, 32.0, 26.0};

  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  SpMv(A, v, r);

  expect_eq_vectorf(r, rexpected, n, tol);

  free(A);
}

TEST(linalg, SpMtv_test)
{
  constexpr QOCOInt m = 5;
  constexpr QOCOInt n = 3;
  QOCOFloat Ax[] = {1, 4, 10, 3, 2, 8, 11, 4, 3, 6, 9, 5};
  QOCOInt Annz = 12;
  QOCOInt Ap[] = {0, 4, 8, 12};
  QOCOInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QOCOFloat v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  QOCOFloat r[n];
  QOCOFloat rexpected[] = {64.0, 90.0, 67.0};

  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  SpMtv(A, v, r);

  expect_eq_vectorf(r, rexpected, n, tol);

  free(A);
}

TEST(linalg, inf_norm_test)
{
  constexpr QOCOInt n = 4;
  QOCOFloat x[] = {-1.5, 6.0, -10.0, 8.0};
  QOCOFloat expected_ans = 10.0;
  QOCOFloat tol = 1e-12;

  EXPECT_NEAR(inf_norm(x, n), expected_ans, tol);
}

TEST(linalg, regularize_test1)
{
  QOCOInt n = 6;
  QOCOFloat Px[] = {1, 2, 3, 4, 5, 6};
  QOCOInt Pnnz = 6;
  QOCOInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QOCOInt Pi[] = {0, 1, 2, 3, 4, 5};

  QOCOFloat Px_exp[] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* Pexp = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(Pexp, n, n, Pnnz, Px_exp, Pp, Pi);

  QOCOCscMatrix* Pmalloc = new_qoco_csc_matrix(P);

  QOCOInt num_diagP = count_diag(Pmalloc);
  QOCOCscMatrix* Pres = regularize_P(num_diagP, Pmalloc, 0.5, NULL);

  expect_eq_csc(Pres, Pexp, tol);

  free(P);
  free(Pexp);
  free_qoco_csc_matrix(Pres);
}

TEST(linalg, regularize_test2)
{
  constexpr QOCOInt n = 3;
  QOCOFloat Px[] = {2, 3};
  constexpr QOCOInt Pnnz = 2;
  QOCOInt Pp[] = {0, 0, 2, 2};
  QOCOInt Pi[] = {0, 1};

  QOCOFloat Px_exp[] = {1, 2, 4, 1};
  QOCOInt Pnnz_exp = 4;
  QOCOInt Pp_exp[] = {0, 1, 3, 4};
  QOCOInt Pi_exp[] = {0, 0, 1, 2};

  QOCOInt* nzadded_idx = (QOCOInt*)malloc(sizeof(QOCOInt) * n);
  QOCOInt nzadded_idx_exp[] = {0, 3};
  QOCOInt nz_added_exp = 2;

  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* Pexp = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(Pexp, n, n, Pnnz_exp, Px_exp, Pp_exp, Pi_exp);

  QOCOCscMatrix* Pmalloc = new_qoco_csc_matrix(P);
  QOCOCscMatrix* Pexpmalloc = new_qoco_csc_matrix(Pexp);

  QOCOInt num_diagP = count_diag(Pmalloc);
  QOCOCscMatrix* Pres = regularize_P(num_diagP, Pmalloc, 1.0, nzadded_idx);

  expect_eq_csc(Pres, Pexpmalloc, tol);

  EXPECT_EQ(n - num_diagP, nz_added_exp);

  for (QOCOInt i = 0; i < n - num_diagP; ++i) {
    EXPECT_EQ(nzadded_idx[i], nzadded_idx_exp[i]);
  }

  free(P);
  free(Pexp);
  free(nzadded_idx);
  free_qoco_csc_matrix(Pres);
  free_qoco_csc_matrix(Pexpmalloc);
}

TEST(linalg, regularize_test3)
{
  QOCOInt n = 6;
  QOCOFloat Px[] = {1, 2, 3, 3, 5, 7, 8};
  QOCOInt Pnnz = 7;
  QOCOInt Pp[] = {0, 1, 3, 4, 5, 7, 7};
  QOCOInt Pi[] = {0, 0, 1, 0, 2, 2, 4};

  QOCOFloat Pxexp[] = {1.001, 2.0,   3.001, 3.0,   0.001,
                       5.0,   0.001, 7.0,   8.001, 0.001};
  QOCOInt Pnnzexp = 10;
  QOCOInt Ppexp[] = {0, 1, 3, 5, 7, 9, 10};
  QOCOInt Piexp[] = {0, 0, 1, 0, 2, 2, 3, 2, 4, 5};
  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* Pexp = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(Pexp, n, n, Pnnzexp, Pxexp, Ppexp, Piexp);

  QOCOCscMatrix* Pmalloc = new_qoco_csc_matrix(P);

  QOCOInt num_diagP = count_diag(Pmalloc);
  QOCOCscMatrix* Pres = regularize_P(num_diagP, Pmalloc, 1e-3, NULL);

  expect_eq_csc(Pres, Pexp, tol);

  free(P);
  free(Pexp);
  free_qoco_csc_matrix(Pres);
}

TEST(linalg, col_inf_norm_USymm_test)
{
  constexpr QOCOInt m = 3;
  constexpr QOCOInt n = 3;
  QOCOFloat Ax[] = {1, 2, 5, 3, 6, 8};
  QOCOInt Annz = 6;
  QOCOInt Ap[] = {0, 1, 3, 6};
  QOCOInt Ai[] = {0, 0, 1, 0, 1, 2};
  QOCOFloat norm[n];
  QOCOFloat norm_expected[] = {3, 6, 8};

  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  col_inf_norm_USymm(A, norm);

  expect_eq_vectorf(norm, norm_expected, n, tol);

  free(A);
}

TEST(linalg, row_inf_norm_test)
{
  constexpr QOCOInt m = 5;
  constexpr QOCOInt n = 3;
  QOCOFloat Ax[] = {1, 4, 10, 3, 2, 8, -11, 4, 3, -6, 9, 5};
  QOCOInt Annz = 12;
  QOCOInt Ap[] = {0, 4, 8, 12};
  QOCOInt Ai[] = {0, 1, 3, 4, 0, 2, 3, 4, 0, 1, 2, 4};
  QOCOFloat norm[m];
  QOCOFloat norm_expected[] = {3.0, 6.0, 9.0, 11.0, 5.0};

  QOCOFloat tol = 1e-12;

  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  qoco_set_csc(A, m, n, Annz, Ax, Ap, Ai);

  row_inf_norm(A, norm);

  expect_eq_vectorf(norm, norm_expected, n, tol);

  free(A);
}

TEST(linalg, ruiz_test)
{
  QOCOInt p = 2;
  QOCOInt m = 6;
  QOCOInt n = 6;
  QOCOInt l = 3;
  QOCOInt nsoc = 1;

  QOCOFloat Px[] = {1, 2, 3, 4, 5, 6};
  QOCOInt Pnnz = 6;
  QOCOInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QOCOInt Pi[] = {0, 1, 2, 3, 4, 5};

  QOCOFloat Ax[] = {1, 1, 1, 2};
  QOCOInt Annz = 4;
  QOCOInt Ap[] = {0, 1, 3, 4, 4, 4, 4};
  QOCOInt Ai[] = {0, 0, 1, 1};

  QOCOFloat Gx[] = {-1, -1, -1, -1, -1, -1};
  QOCOInt Gnnz = 6;
  QOCOInt Gp[] = {0, 1, 2, 3, 4, 5, 6};
  QOCOInt Gi[] = {0, 1, 2, 3, 4, 5};

  QOCOFloat c[] = {1, 2, 3, 4, 5, 6};
  QOCOFloat b[] = {1, 2};
  QOCOFloat h[] = {0, 0, 0, 0, 0, 0};
  QOCOInt q[] = {3};

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOFloat Dexp[] = {1.0000, 0.8409, 0.6389, 0.7071, 0.7022, 0.6894};
  QOCOFloat Eexp[] = {1.0000, 0.7825};
  QOCOFloat Fexp[] = {1.0000, 1.1892, 1.5315, 1.4142, 1.4142, 1.4142};
  QOCOFloat kexp = 0.2480;
  QOCOFloat tol = 1e-4;

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  settings->ruiz_iters = 5;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

  qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

  expect_eq_vectorf(get_data_vectorf(solver->work->scaling->Druiz), Dexp, n, tol);
  expect_eq_vectorf(get_data_vectorf(solver->work->scaling->Eruiz), Eexp, p, tol);
  expect_eq_vectorf(get_data_vectorf(solver->work->scaling->Fruiz), Fexp, m, tol);
  EXPECT_NEAR(solver->work->scaling->k, kexp, tol);

  qoco_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}
