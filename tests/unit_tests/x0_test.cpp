#include "qoco.h"
#include "gtest/gtest.h"

#include <cstdlib>

namespace {

void set_single_iteration_accept_settings(QOCOSettings* settings)
{
  set_default_settings(settings);
  settings->max_iters = 1;
  settings->ruiz_iters = 0;
  settings->abstol = 1e20;
  settings->reltol = 1e20;
  settings->verbose = 0;
}

QOCOSolver* setup_unconstrained_qp(QOCOFloat c0, QOCOFloat c1,
                                   QOCOInt ruiz_iters = 0)
{
  QOCOInt n = 2;
  QOCOInt m = 0;
  QOCOInt p = 0;
  QOCOInt l = 0;
  QOCOInt nsoc = 0;

  QOCOFloat Px[] = {1, 1};
  QOCOInt Pp[] = {0, 1, 2};
  QOCOInt Pi[] = {0, 1};
  QOCOFloat c[] = {c0, c1};
  QOCOCscMatrix P;
  qoco_set_csc(&P, n, n, 2, Px, Pp, Pi);

  QOCOSettings settings;
  set_single_iteration_accept_settings(&settings);
  settings.ruiz_iters = ruiz_iters;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, &P, c, nullptr, nullptr, nullptr,
                            nullptr, l, nsoc, nullptr, &settings);
  EXPECT_EQ(exit, QOCO_NO_ERROR);
  if (exit != QOCO_NO_ERROR) {
    free(solver);
    return nullptr;
  }
  return solver;
}

void expect_near_array(const QOCOFloat* actual, const QOCOFloat* expected,
                       QOCOInt n, QOCOFloat tol)
{
  for (QOCOInt i = 0; i < n; ++i) {
    EXPECT_NEAR(actual[i], expected[i], tol);
  }
}

} // namespace

TEST(x0, custom_primal_start_uses_original_scaling)
{
  QOCOSolver* solver = setup_unconstrained_qp(1, 1, 1);
  ASSERT_NE(solver, nullptr);

  QOCOFloat x0[] = {2.5, -3.0};
  qoco_set_x0(solver, x0);
  QOCOInt exit = qoco_solve(solver);

  ASSERT_EQ(exit, QOCO_SOLVED);
  expect_near_array(solver->sol->x, x0, 2, 1e-6);

  qoco_cleanup(solver);
}

TEST(x0, custom_primal_start_is_copied)
{
  QOCOSolver* solver = setup_unconstrained_qp(1, 2);
  ASSERT_NE(solver, nullptr);

  QOCOFloat x0[] = {2.0, -3.0};
  QOCOFloat x0_expected[] = {2.0, -3.0};
  qoco_set_x0(solver, x0);
  x0[0] = 99.0;
  x0[1] = 99.0;

  QOCOInt exit = qoco_solve(solver);

  ASSERT_EQ(exit, QOCO_SOLVED);
  expect_near_array(solver->sol->x, x0_expected, 2, 1e-6);

  qoco_cleanup(solver);
}

TEST(x0, custom_primal_start_can_be_cleared)
{
  QOCOSolver* solver = setup_unconstrained_qp(1, 2);
  ASSERT_NE(solver, nullptr);

  QOCOFloat x0[] = {2.0, -3.0};
  qoco_set_x0(solver, x0);
  qoco_set_x0(solver, nullptr);

  QOCOInt exit = qoco_solve(solver);

  QOCOFloat default_start[] = {-1.0, -2.0};
  ASSERT_EQ(exit, QOCO_SOLVED);
  expect_near_array(solver->sol->x, default_start, 2, 1e-6);

  qoco_cleanup(solver);
}

TEST(x0, custom_primal_start_sets_consistent_linear_cone_slack)
{
  QOCOInt n = 2;
  QOCOInt m = 3;
  QOCOInt p = 0;
  QOCOInt l = 3;
  QOCOInt nsoc = 0;

  QOCOFloat Gx[] = {-1, 1, -1, 1};
  QOCOInt Gp[] = {0, 2, 4};
  QOCOInt Gi[] = {0, 2, 1, 2};
  QOCOFloat c[] = {-1, -2};
  QOCOFloat h[] = {0, 0, 1};
  QOCOCscMatrix G;
  qoco_set_csc(&G, m, n, 4, Gx, Gp, Gi);

  QOCOSettings settings;
  set_single_iteration_accept_settings(&settings);

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, nullptr, c, nullptr, nullptr, &G,
                            h, l, nsoc, nullptr, &settings);
  ASSERT_EQ(exit, QOCO_NO_ERROR);

  QOCOFloat x0[] = {0.25, 0.5};
  QOCOFloat s0_expected[] = {0.25, 0.5, 0.25};
  qoco_set_x0(solver, x0);
  exit = qoco_solve(solver);

  ASSERT_EQ(exit, QOCO_SOLVED);
  expect_near_array(solver->sol->x, x0, n, 1e-6);
  expect_near_array(solver->sol->s, s0_expected, m, 1e-6);

  qoco_cleanup(solver);
}
