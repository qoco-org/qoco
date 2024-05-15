#include "test_utils.h"
#include "gtest/gtest.h"

#include "qcos.h"

TEST(input_validation, settings_validation)
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

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  set_default_settings(settings);
  settings->max_iter_bisection = 0;
  QCOSInt exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);
  settings->max_iter_bisection = -1;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->max_iters = 0;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);
  settings->max_iters = -1;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->abstol = 0;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);
  settings->abstol = -1e-6;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->reltol = -1e-6;
  exit = qcos_setup(solver, P, c, A, b, G, h, l, ncones, q, settings);
  EXPECT_EQ(exit, QCOS_SETTINGS_VALIDATION_ERROR);

  free(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}