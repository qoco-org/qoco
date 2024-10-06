#include "test_utils.h"
#include "gtest/gtest.h"

#include "qcos.h"

TEST(simple_socp_test, p1)
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
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
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
  QCOSInt nsoc = 2;

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
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
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
  QCOSInt nsoc = 2;

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
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}

TEST(simple_socp_test, TAME)
{
  QCOSInt p = 1;
  QCOSInt m = 2;
  QCOSInt n = 2;
  QCOSInt l = 2;
  QCOSInt nsoc = 0;

  QCOSFloat Px[] = {2, -2, 2};
  QCOSInt Pnnz = 3;
  QCOSInt Pp[] = {0, 1, 3};
  QCOSInt Pi[] = {0, 0, 1};

  QCOSFloat Ax[] = {1, 1};
  QCOSInt Annz = 2;
  QCOSInt Ap[] = {0, 1, 2};
  QCOSInt Ai[] = {0, 0};

  QCOSFloat Gx[] = {-1, -1};
  QCOSInt Gnnz = 2;
  QCOSInt Gp[] = {0, 1, 2};
  QCOSInt Gi[] = {0, 1};

  QCOSFloat c[] = {0, 0};
  QCOSFloat b[] = {1};
  QCOSFloat h[] = {0, 0};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, nullptr, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  QCOSFloat tol = 1e-8;
  QCOSFloat xexp[] = {0.5, 0.5};
  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  ASSERT_NEAR(solver->sol->obj, 0, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}

TEST(simple_socp_test, reduced_tolerance)
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

  QCOSFloat xexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat sexp[] = {0.2000, 0.8000, 0.6000, 0.3981, -0.2625, -0.2993};
  QCOSFloat yexp[] = {-1.200, -2.400};
  QCOSFloat zexp[] = {0.0000, 0.0000, 0.0000, 5.5923, 3.6875, 4.2043};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->abstol = 1e-12;
  settings->reltol = 1e-12;
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED_INACCURATE);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}

TEST(simple_socp_test, update_vector_data_test)
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

  QCOSFloat cnew[] = {0, 0, 0, 0, 0, 0};
  QCOSFloat bnew[] = {4, 5};
  QCOSFloat hnew[] = {1, 1, 1, 1, 1, 1};

  QCOSInt q[] = {3};

  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSFloat xexp[] = {1.9333, 2.0667, 1.4667, 0.2391, -0.1337, -0.1140};
  QCOSFloat sexp[] = {2.9333, 3.0667, 2.4667, 1.2391, 0.8663, 0.8860};
  QCOSFloat yexp[] = {-1.9333, -2.2000};
  QCOSFloat zexp[] = {0.0000, 0.0000, 0.0000, 0.9565, -0.6687, -0.6839};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  update_vector_data(solver, cnew, bnew, hnew);
  exit = qcos_solve(solver);

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}

TEST(simple_socp_test, update_constraint_data_test)
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
  QCOSFloat Axnew[] = {1, 2, 3, 4};
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

  QCOSFloat xexp[] = {0.0000, 0.5000, 0.1250, 0.3981, -0.2625, -0.2993};
  QCOSFloat sexp[] = {0.0000, 0.5000, 0.1250, 0.3981, -0.2625, -0.2993};
  QCOSFloat yexp[] = {-0.2344, -0.8437};
  QCOSFloat zexp[] = {0.7656, 0.0000, 0.0000, 5.5923, 3.6875, 4.2043};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  update_matrix_data(solver, NULL, Axnew, NULL);

  exit = qcos_solve(solver);

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}

TEST(simple_socp_test, update_cost_matrix_test)
{
  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 3;
  QCOSInt nsoc = 1;

  QCOSFloat Px[] = {1, 1, 3, 5, 8};
  QCOSFloat Pxnew[] = {2, 1, 4, 5, 8};

  QCOSInt Pnnz = 5;
  QCOSInt Pp[] = {0, 1, 3, 3, 4, 5, 5};
  QCOSInt Pi[] = {0, 0, 1, 3, 4};

  QCOSFloat Ax[] = {1, 1, 1, 2};
  QCOSFloat Axnew[] = {1, 2, 3, 4};
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

  QCOSFloat xexp[] = {0.0000, 0.5000, 0.1250, 0.5447, -0.2458, -0.4861};
  QCOSFloat sexp[] = {0.0000, 0.5000, 0.1250, 0.5447, -0.2458, -0.4861};
  QCOSFloat yexp[] = {-0.8750, -0.7500};
  QCOSFloat zexp[] = {0.6250, 0.0000, 0.0000, 6.7234, 3.0338, 6.0000};
  QCOSFloat tol = 1e-4;

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  update_matrix_data(solver, Pxnew, Axnew, NULL);

  exit = qcos_solve(solver);

  expect_eq_vectorf(solver->sol->x, xexp, n, tol);
  expect_eq_vectorf(solver->sol->s, sexp, m, tol);
  expect_eq_vectorf(solver->sol->y, yexp, p, tol);
  expect_eq_vectorf(solver->sol->z, zexp, n, tol);
  ASSERT_EQ(exit, QCOS_SOLVED);

  qcos_cleanup(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}