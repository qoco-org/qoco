#include "test_utils.h"
#include "gtest/gtest.h"

#include "qoco.h"

TEST(input_validation, settings_validation)
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

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

  set_default_settings(settings);
  settings->bisect_iters = 0;
  QOCOInt exit =
      qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);
  settings->bisect_iters = -1;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->max_iters = 0;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);
  settings->max_iters = -1;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->abstol = 0;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);
  settings->abstol = -1e-6;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);

  set_default_settings(settings);
  settings->reltol = -1e-6;
  exit = qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  EXPECT_EQ(exit, QOCO_SETTINGS_VALIDATION_ERROR);

  free(solver);
  free(settings);
  free(P);
  free(A);
  free(G);
}