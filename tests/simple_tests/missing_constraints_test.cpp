#include "test_utils.h"
#include "gtest/gtest.h"

#include "qcos.h"

TEST(missing_constraints_test, no_soc_constraints)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 6;
  QCOSInt nsoc = 0;

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

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {0.2000, 0.8000, 0.6000, -0.0000, 0.0000, 0.0000};
  QCOSFloat sexp[] = {0.2000, 0.8000, 0.6000, 0.0000, 0.0000, 0.0000};
  QCOSFloat yexp[] = {-1.200, -2.400};
  QCOSFloat zexp[] = {0.0000, 0.0000, 0.0000, 4.0000, 5.0000, 6.0000};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, nullptr, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(P);
  free(A);
  free(G);
}

TEST(missing_constraints_test, no_ineq_constraints)
{
  QCOSInt m = 0;
  QCOSInt p = 2;
  QCOSInt n = 6;
  QCOSInt l = 0;
  QCOSInt nsoc = 0;

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat Ax[] = {1, 1, 1, 2};
  QCOSInt Annz = 4;
  QCOSInt Ap[] = {0, 1, 3, 4, 4, 4, 4};
  QCOSInt Ai[] = {0, 0, 1, 1};

  QCOSFloat c[] = {1, 2, 3, 4, 5, 6};
  QCOSFloat b[] = {1, 2};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);

  QCOSFloat xexp[] = {0.2000, 0.8000, 0.6000, -1.0000, -1.0000, -1.0000};
  QCOSFloat yexp[] = {-1.200, -2.400};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit = qcos_setup(solver, n, m, p, P, c, A, b, nullptr, nullptr, l,
                            nsoc, nullptr, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(P);
  free(A);
}

TEST(missing_constraints_test, no_eq_constraints)
{
  QCOSInt m = 6;
  QCOSInt p = 0;
  QCOSInt n = 6;
  QCOSInt l = 3;
  QCOSInt nsoc = 1;

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat Gx[] = {-1, -1, -1, -1, -1, -1};
  QCOSInt Gnnz = 6;
  QCOSInt Gp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Gi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat c[] = {1, 2, 3, 4, 5, 6};
  QCOSFloat h[] = {0, 0, 0, 0, 0, 0};
  QCOSInt q[] = {3};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {0.0000, -0.0000, -0.0000, 0.3981, -0.2624, -0.2993};
  QCOSFloat sexp[] = {0.0000, 0.0000, 0.0000, 0.3981, -0.2624, -0.2993};
  QCOSFloat zexp[] = {1.0000, 2.0000, 3.0000, 5.5923, 3.6878, 4.2040};
  QCOSFloat tol = 1e-3;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit = qcos_setup(solver, n, m, p, P, c, nullptr, nullptr, G, h, l,
                            nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(P);
  free(G);
}

TEST(missing_constraints_test, no_constraints)
{
  QCOSInt m = 0;
  QCOSInt p = 0;
  QCOSInt l = 0;
  QCOSInt n = 6;
  QCOSInt nsoc = 0;

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};
  QCOSFloat c[] = {1, 2, 3, 4, 5, 6};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);

  QCOSFloat xexp[] = {-1, -1, -1, -1, -1, -1};
  QCOSFloat tol = 1e-3;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit = qcos_setup(solver, n, m, p, P, c, nullptr, nullptr, nullptr,
                            nullptr, l, nsoc, nullptr, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(P);
}

TEST(missing_constraints_test, lp_test_no_P)
{
  //  Solves the following LP
  //  minimize   -x1 - x2
  //  subject to x1 >= 0
  //             x2 >= 0
  //             x1 + x2 <= 1

  QCOSInt p = 0;
  QCOSInt m = 3;
  QCOSInt n = 2;
  QCOSInt l = 3;
  QCOSInt nsoc = 0;

  QCOSFloat Gx[] = {-1, 1, -1, 1};
  QCOSInt Gnnz = 4;
  QCOSInt Gp[] = {0, 2, 4};
  QCOSInt Gi[] = {0, 2, 1, 2};

  QCOSFloat c[] = {-1, -2};
  QCOSFloat h[] = {0, 0, 1};

  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {0.0, 1.0};
  QCOSFloat sexp[] = {0.0, 1.0, 0.0};
  QCOSFloat zexp[] = {1.0, 0.0, 2.0};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit = qcos_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, G, h,
                            l, nsoc, nullptr, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(G);
}
