#include "test_utils.h"
#include "gtest/gtest.h"

#include "qcos.h"

TEST(simple_socp_test, p1)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 3;
  QCOSInt ncones = 1;

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

  QCOSFloat xexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat sexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat yexp[] = {-1.200, -2.400};
  QCOSFloat zexp[] = {0.0000, 0.0000, 0.0000, 5.5923, 3.6875, 4.2043};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, ncones, q, settings);
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

TEST(simple_socp_test, p2)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 1;
  QCOSInt ncones = 2;

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
  QCOSInt q[] = {2, 3};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat sexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat yexp[] = {-1.2000, -2.4000};
  QCOSFloat zexp[] = {0.0000, 0.0000, -0.0000, 5.5923, 3.6876, 4.2043};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, ncones, q, settings);
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

TEST(simple_socp_test, p3)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 0;
  QCOSInt ncones = 2;

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
  QCOSInt q[] = {3, 3};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {1.0000, 0.0000, 1.0000, 0.3981, -0.2625, -0.2993};
  QCOSFloat sexp[] = {1.0000, 0.0000, 1.0000, 0.3981, -0.2625, -0.2993};
  QCOSFloat yexp[] = {4.0000, -6.0000};
  QCOSFloat zexp[] = {6.0000, -0.0000, -6.0000, 5.5923, 3.6876, 4.2043};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, ncones, q, settings);
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