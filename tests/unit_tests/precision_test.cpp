#include "qoco.h"
#include "qdldl.h"
#include "gtest/gtest.h"

#include <float.h>

namespace {

QOCOFloat precise_one()
{
#ifdef QOCO_LONG_DOUBLE_PRECISION
  return (QOCOFloat)(1.0L + LDBL_EPSILON);
#else
  return (QOCOFloat)1.0;
#endif
}

} // namespace

TEST(precision, qoco_and_qdldl_float_types_match)
{
  EXPECT_EQ(sizeof(QOCOFloat), sizeof(QDLDL_float));

#ifdef QOCO_SINGLE_PRECISION
  EXPECT_EQ(sizeof(QOCOFloat), sizeof(float));
#elif defined(QOCO_LONG_DOUBLE_PRECISION)
  EXPECT_GT(LDBL_MANT_DIG, DBL_MANT_DIG);
  EXPECT_EQ(sizeof(QOCOFloat), sizeof(long double));
  EXPECT_EQ((double)precise_one(), 1.0);
  EXPECT_NE(precise_one(), (QOCOFloat)(double)precise_one());
#else
  EXPECT_EQ(sizeof(QOCOFloat), sizeof(double));
#endif
}

TEST(precision, vector_and_matrix_copies_preserve_input_precision)
{
  QOCOFloat x[] = {precise_one()};
  QOCOFloat Ax[] = {precise_one()};
  QOCOInt Ap[] = {0, 1};
  QOCOInt Ai[] = {0};
  QOCOCscMatrix A;
  qoco_set_csc(&A, 1, 1, 1, Ax, Ap, Ai);

  QOCOVectorf* v = new_qoco_vectorf(x, 1);
  QOCOCscMatrix* M = new_qoco_csc_matrix(&A);

  EXPECT_EQ(get_element_vectorf(v, 0), x[0]);
  EXPECT_EQ(M->x[0], Ax[0]);
#ifdef QOCO_LONG_DOUBLE_PRECISION
  EXPECT_NE(get_element_vectorf(v, 0), (QOCOFloat)(double)x[0]);
  EXPECT_NE(M->x[0], (QOCOFloat)(double)Ax[0]);
#endif

  free_qoco_vectorf(v);
  free_qoco_csc_matrix(M);
}

TEST(precision, setup_and_solve_do_not_mutate_user_input)
{
  QOCOInt n = 1;
  QOCOInt m = 0;
  QOCOInt p = 0;
  QOCOInt l = 0;
  QOCOInt nsoc = 0;

  QOCOFloat Px[] = {precise_one()};
  QOCOFloat c[] = {precise_one()};
  QOCOInt Pp[] = {0, 1};
  QOCOInt Pi[] = {0};
  QOCOFloat Px_orig[] = {Px[0]};
  QOCOFloat c_orig[] = {c[0]};

  QOCOCscMatrix P;
  qoco_set_csc(&P, n, n, 1, Px, Pp, Pi);

  QOCOSettings settings;
  set_default_settings(&settings);
  settings.verbose = 0;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit = qoco_setup(solver, n, m, p, &P, c, nullptr, nullptr, nullptr,
                            nullptr, l, nsoc, nullptr, &settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  EXPECT_EQ(Px[0], Px_orig[0]);
  EXPECT_EQ(c[0], c_orig[0]);
#ifdef QOCO_LONG_DOUBLE_PRECISION
  EXPECT_NE(Px[0], (QOCOFloat)(double)Px_orig[0]);
  EXPECT_NE(c[0], (QOCOFloat)(double)c_orig[0]);
#endif
  EXPECT_EQ(exit, QOCO_SOLVED);

  qoco_cleanup(solver);
}
