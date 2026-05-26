#include "gtest/gtest.h"

#include "qoco.h"

#ifdef QOCO_ALGEBRA_BACKEND_CUDA
static bool batch_status_ok(QOCOInt status)
{
  return status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE;
}
#endif

#ifndef QOCO_ALGEBRA_BACKEND_CUDA
TEST(batch_api, unsupported_on_builtin_backend)
{
  QOCOBatchSolver batch;
  QOCOSettings settings;
  set_default_settings(&settings);

  QOCOFloat c[] = {1.0};
  QOCOFloat h[] = {0.0, 1.0};
  QOCOFloat Gx[] = {-1.0, 1.0};
  QOCOInt Gi[] = {0, 1};
  QOCOInt Gp[] = {0, 2};
  QOCOCscMatrix G;
  qoco_set_csc(&G, 2, 1, 2, Gx, Gp, Gi);

  QOCOInt exit = qoco_batch_setup(&batch, 2, 1, 2, 0, nullptr, c, nullptr,
                                  nullptr, &G, h, 2, 0, nullptr, &settings);

  EXPECT_EQ(exit, QOCO_UNSUPPORTED_ERROR);
}
#else
TEST(batch_api, solves_same_sparsity_cuda_batch)
{
  QOCOBatchSolver batch;
  QOCOSettings settings;
  set_default_settings(&settings);
  settings.verbose = 0;

  QOCOFloat c[] = {1.0};
  QOCOFloat h[] = {0.0, 1.0};
  QOCOFloat Gx[] = {-1.0, 1.0};
  QOCOInt Gi[] = {0, 1};
  QOCOInt Gp[] = {0, 2};
  QOCOCscMatrix G;
  qoco_set_csc(&G, 2, 1, 2, Gx, Gp, Gi);

  QOCOInt exit = qoco_batch_setup(&batch, 2, 1, 2, 0, nullptr, c, nullptr,
                                  nullptr, &G, h, 2, 0, nullptr, &settings);
  ASSERT_EQ(exit, QOCO_NO_ERROR);

  QOCOFloat c1[] = {-1.0};
  QOCOFloat h1[] = {0.0, 1.0};
  QOCOFloat Gx1[] = {-1.0, 0.5};
  EXPECT_EQ(qoco_batch_update_vector_data(&batch, 1, c1, nullptr, h1),
            QOCO_NO_ERROR);
  EXPECT_EQ(qoco_batch_update_matrix_data(&batch, 1, nullptr, nullptr, Gx1),
            QOCO_NO_ERROR);
  EXPECT_EQ(qoco_batch_update_vector_data(&batch, 2, c1, nullptr, h1),
            QOCO_DATA_VALIDATION_ERROR);

  exit = qoco_batch_solve(&batch);
  ASSERT_EQ(exit, QOCO_NO_ERROR);
  ASSERT_TRUE(batch_status_ok(batch.statuses[0]));
  ASSERT_TRUE(batch_status_ok(batch.statuses[1]));

  QOCOSolution* sol0 = qoco_batch_get_solution(&batch, 0);
  QOCOSolution* sol1 = qoco_batch_get_solution(&batch, 1);
  ASSERT_NE(sol0, nullptr);
  ASSERT_NE(sol1, nullptr);
  EXPECT_NEAR(sol0->x[0], 0.0, 1e-3);
  EXPECT_NEAR(sol1->x[0], 2.0, 1e-3);
  EXPECT_EQ(qoco_batch_get_solution(&batch, 2), nullptr);

  EXPECT_EQ(qoco_batch_cleanup(&batch), QOCO_NO_ERROR);
}
#endif
