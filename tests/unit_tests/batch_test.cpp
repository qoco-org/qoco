#include "qoco.h"
#include "test_utils.h"
#include "gtest/gtest.h"

#include <cstdlib>

namespace {

void set_batch_test_settings(QOCOSettings* settings)
{
  set_default_settings(settings);
  settings->ruiz_iters = 0;
  settings->verbose = 0;
}

void set_diag_qp_matrix(QOCOCscMatrix* P, QOCOFloat* Px)
{
  static QOCOInt Pp[] = {0, 1, 2};
  static QOCOInt Pi[] = {0, 1};
  qoco_set_csc(P, 2, 2, 2, Px, Pp, Pi);
}

QOCOSolver* setup_serial(QOCOFloat* Px, QOCOFloat* c,
                         QOCOSettings* settings)
{
  QOCOCscMatrix P;
  set_diag_qp_matrix(&P, Px);

  QOCOSolver* solver = (QOCOSolver*)std::malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, 2, 0, 0, &P, c, nullptr, nullptr, nullptr,
                            nullptr, 0, 0, nullptr, settings);
  EXPECT_EQ(exit, QOCO_NO_ERROR);
  if (exit != QOCO_NO_ERROR) {
    std::free(solver);
    return nullptr;
  }
  return solver;
}

} // namespace

TEST(batch, vector_and_matrix_updates_match_serial_solves)
{
  QOCOFloat Px_base[] = {1.0, 2.0};
  QOCOFloat c_base[] = {1.0, 4.0};
  QOCOFloat c_item0[] = {2.0, -6.0};
  QOCOFloat Px_item1[] = {2.0, 4.0};

  QOCOSettings settings;
  set_batch_test_settings(&settings);

  QOCOCscMatrix P;
  set_diag_qp_matrix(&P, Px_base);

  QOCOBatchSolver batch;
  QOCOInt exit = qoco_batch_setup(&batch, 2, 2, 0, 0, &P, c_base, nullptr,
                                  nullptr, nullptr, nullptr, 0, 0, nullptr,
                                  &settings);
  ASSERT_EQ(exit, QOCO_NO_ERROR);

  EXPECT_EQ(qoco_batch_update_vector_data(&batch, 0, c_item0, nullptr, nullptr),
            QOCO_NO_ERROR);
  EXPECT_EQ(qoco_batch_update_matrix_data(&batch, 1, Px_item1, nullptr,
                                          nullptr),
            QOCO_NO_ERROR);

  QOCOSolver* serial0 = setup_serial(Px_base, c_base, &settings);
  ASSERT_NE(serial0, nullptr);
  qoco_update_vector_data(serial0, c_item0, nullptr, nullptr);
  QOCOInt serial0_status = qoco_solve(serial0);

  QOCOSolver* serial1 = setup_serial(Px_base, c_base, &settings);
  ASSERT_NE(serial1, nullptr);
  qoco_update_matrix_data(serial1, Px_item1, nullptr, nullptr);
  QOCOInt serial1_status = qoco_solve(serial1);

  exit = qoco_batch_solve(&batch);
  ASSERT_EQ(exit, QOCO_NO_ERROR);
  ASSERT_EQ(batch.statuses[0], serial0_status);
  ASSERT_EQ(batch.statuses[1], serial1_status);

  QOCOSolution* sol0 = qoco_batch_get_solution(&batch, 0);
  QOCOSolution* sol1 = qoco_batch_get_solution(&batch, 1);
  ASSERT_NE(sol0, nullptr);
  ASSERT_NE(sol1, nullptr);

  expect_eq_vectorf(sol0->x, serial0->sol->x, 2, 1e-6);
  expect_eq_vectorf(sol1->x, serial1->sol->x, 2, 1e-6);
  EXPECT_NEAR((double)sol0->obj, (double)serial0->sol->obj, 1e-6);
  EXPECT_NEAR((double)sol1->obj, (double)serial1->sol->obj, 1e-6);

  qoco_cleanup(serial0);
  qoco_cleanup(serial1);
  qoco_batch_cleanup(&batch);
}

TEST(batch, batch_count_one_matches_serial_solve)
{
  QOCOFloat Px[] = {1.0, 2.0};
  QOCOFloat c[] = {-3.0, 4.0};

  QOCOSettings settings;
  set_batch_test_settings(&settings);

  QOCOCscMatrix P;
  set_diag_qp_matrix(&P, Px);

  QOCOBatchSolver batch;
  QOCOInt exit = qoco_batch_setup(&batch, 1, 2, 0, 0, &P, c, nullptr, nullptr,
                                  nullptr, nullptr, 0, 0, nullptr, &settings);
  ASSERT_EQ(exit, QOCO_NO_ERROR);

  QOCOSolver* serial = setup_serial(Px, c, &settings);
  ASSERT_NE(serial, nullptr);
  QOCOInt serial_status = qoco_solve(serial);

  exit = qoco_batch_solve(&batch);
  ASSERT_EQ(exit, QOCO_NO_ERROR);
  ASSERT_EQ(batch.statuses[0], serial_status);

  QOCOSolution* sol = qoco_batch_get_solution(&batch, 0);
  ASSERT_NE(sol, nullptr);
  expect_eq_vectorf(sol->x, serial->sol->x, 2, 1e-6);
  EXPECT_EQ(qoco_batch_get_solution(&batch, 1), nullptr);

  qoco_cleanup(serial);
  qoco_batch_cleanup(&batch);
}
